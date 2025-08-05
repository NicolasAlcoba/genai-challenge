import os
from typing import Dict, List, Tuple
from nltk.tokenize import sent_tokenize

from .base_chunk import BaseChunker, DocumentSection
from .semantic_chunk import SemanticChunker


class HybridChunker(BaseChunker):
    """Advanced chunking that adapts strategy based on content characteristics"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Initialize semantic analyzer for topic detection
        self.semantic_analyzer = SemanticChunker(**kwargs)
        
    def chunk_sections(self, sections: List[DocumentSection], source: str) -> List[Dict[str, str]]:
        """Apply hybrid chunking strategy"""
        chunks = []
        chunk_id = 0
        
        # First, analyze the document structure
        doc_analysis = self._analyze_document_structure(sections)
        
        # Group sections intelligently based on multiple factors
        section_groups = self._create_intelligent_groups(sections, doc_analysis)
        
        # Process each group with the most appropriate strategy
        for group_data in section_groups:
            group = group_data['sections']
            strategy = group_data['strategy']
            
            if strategy == 'keep_together':
                # Keep this group as a single chunk
                chunk_text = self._format_group_optimally(group, group_data)
                chunks.append({
                    "text": chunk_text,
                    "page": str(group[0].page_num),
                    "chunk_id": str(chunk_id),
                    "source": os.path.basename(source),
                    "title": self._create_group_title(group),
                    "chunking_method": "hybrid-structural",
                    "sections_count": len(group),
                    "topics": group_data.get('topics', []),
                    "chunk_type": group_data.get('type', 'mixed')
                })
                chunk_id += 1
                
            elif strategy == 'split_sentences':
                # Use sentence-based splitting for this group
                sentence_chunks = self._split_by_sentences_with_context(group, group_data)
                for chunk_data in sentence_chunks:
                    chunks.append({
                        "text": chunk_data['text'],
                        "page": str(chunk_data['page']),
                        "chunk_id": str(chunk_id),
                        "source": os.path.basename(source),
                        "title": chunk_data['title'],
                        "chunking_method": "hybrid-sentence",
                        "topics": group_data.get('topics', []),
                        "chunk_type": group_data.get('type', 'mixed')
                    })
                    chunk_id += 1
                    
            elif strategy == 'split_semantic':
                # Use semantic splitting
                semantic_chunks = self._split_by_semantic_boundaries(group, group_data)
                for chunk_data in semantic_chunks:
                    chunks.append({
                        "text": chunk_data['text'],
                        "page": str(chunk_data['page']),
                        "chunk_id": str(chunk_id),
                        "source": os.path.basename(source),
                        "title": chunk_data['title'],
                        "chunking_method": "hybrid-semantic",
                        "topics": chunk_data.get('topics', []),
                        "chunk_type": group_data.get('type', 'mixed')
                    })
                    chunk_id += 1
        
        return chunks
    
    def _analyze_document_structure(self, sections: List[DocumentSection]) -> Dict:
        """Analyze document characteristics to inform chunking strategy"""
        analysis = {
            'total_sections': len(sections),
            'has_hierarchy': False,
            'avg_section_length': 0,
            'section_length_variance': 0,
            'dominant_section_types': [],
            'content_types': set()
        }
        
        if not sections:
            return analysis
        
        # Check for hierarchical structure
        levels = [s.level for s in sections]
        analysis['has_hierarchy'] = len(set(levels)) > 1
        
        # Calculate section statistics
        lengths = [len(s.content) for s in sections]
        analysis['avg_section_length'] = sum(lengths) / len(lengths)
        
        # Identify content types
        for section in sections:
            content = section.content.lower()
            if any(word in content for word in ['step', 'procedure', '1.', '2.', '3.']):
                analysis['content_types'].add('procedural')
            if any(word in content for word in ['definition', 'means', 'refers to']):
                analysis['content_types'].add('definitional')
            if any(word in content for word in ['example', 'for instance', 'such as']):
                analysis['content_types'].add('examples')
            if len(sent_tokenize(content)) > 5:
                analysis['content_types'].add('narrative')
        
        return analysis
    
    def _create_intelligent_groups(self, sections: List[DocumentSection], 
                                 doc_analysis: Dict) -> List[Dict]:
        """Create groups of sections with appropriate chunking strategies"""
        groups = []
        i = 0
        
        while i < len(sections):
            current_section = sections[i]
            group_info = self._determine_group_boundaries(sections, i, doc_analysis)
            
            group = sections[i:i + group_info['size']]
            
            # Assign topics to the group
            topics = self._get_group_topics(group)
            
            # Determine best strategy for this group
            strategy = self._select_chunking_strategy(group, group_info, topics)
            
            groups.append({
                'sections': group,
                'strategy': strategy,
                'type': group_info['type'],
                'topics': topics,
                'metadata': group_info
            })
            
            i += group_info['size']
        
        return groups
    
    def _determine_group_boundaries(self, sections: List[DocumentSection], 
                                   start_idx: int, doc_analysis: Dict) -> Dict:
        """Determine optimal group boundaries starting from start_idx"""
        current = sections[start_idx]
        group_size = 1
        group_type = self._classify_section_type(current)
        total_size = len(current.content) + (len(current.title) if current.title else 0)
        
        # Look ahead to find related sections
        for j in range(start_idx + 1, len(sections)):
            next_section = sections[j]
            next_size = len(next_section.content) + (len(next_section.title) if next_section.title else 0)
            
            # Check various cohesion factors
            should_group = False
            
            # 1. Structural cohesion (parent-child or siblings)
            if self._are_structurally_related(current, next_section):
                should_group = True
            
            # 2. Semantic cohesion
            elif self._are_semantically_related(current, next_section):
                should_group = True
            
            # 3. Content type cohesion (e.g., continuing procedure)
            elif self._is_continuation(sections[j-1], next_section):
                should_group = True
            
            # 4. Size constraints
            if should_group and total_size + next_size <= self.max_chunk_size * 1.2:
                group_size += 1
                total_size += next_size
                # Update group type if mixed
                next_type = self._classify_section_type(next_section)
                if next_type != group_type:
                    group_type = 'mixed'
            else:
                break
        
        return {
            'size': group_size,
            'type': group_type,
            'total_size': total_size,
            'is_complete_unit': self._is_complete_semantic_unit(sections[start_idx:start_idx + group_size])
        }
    
    def _classify_section_type(self, section: DocumentSection) -> str:
        """Classify the type of content in a section"""
        content = section.content.lower()
        title = section.title.lower() if section.title else ""
        
        # Check for specific content patterns
        if any(word in title + content for word in ['procedure', 'steps', 'how to']):
            return 'procedural'
        elif any(word in content for word in ['warning', 'caution', 'danger']):
            return 'warning'
        elif any(pattern in content for pattern in ['1.', '2.', '•', '-']):
            return 'list'
        elif len(sent_tokenize(section.content)) <= 2:
            return 'definition'
        else:
            return 'narrative'
    
    def _is_continuation(self, prev_section: DocumentSection, 
                        next_section: DocumentSection) -> bool:
        """Check if next section is a continuation of previous"""
        # Check for continuation indicators
        continuation_patterns = [
            r'^(additionally|furthermore|moreover|also)',
            r'^(continued|continuing)',
            r'^\d+\.',  # Numbered lists
            r'^[a-z]\.',  # Lettered lists
        ]
        
        import re
        for pattern in continuation_patterns:
            if re.match(pattern, next_section.content.strip(), re.IGNORECASE):
                return True
        
        # Check if previous section seems incomplete
        if prev_section.content.rstrip().endswith((',', ';', ':')):
            return True
        
        return False
    
    def _is_complete_semantic_unit(self, sections: List[DocumentSection]) -> bool:
        """Check if a group of sections forms a complete semantic unit"""
        if not sections:
            return False
        
        # Check if it's a complete procedure
        content = ' '.join(s.content for s in sections)
        
        # Look for completeness indicators
        has_introduction = any(word in content.lower()[:200] for word in 
                             ['overview', 'introduction', 'this section', 'following'])
        has_conclusion = any(word in content.lower()[-200:] for word in 
                           ['complete', 'finish', 'done', 'end', 'summary'])
        
        # Check for balanced structure (e.g., numbered steps)
        import re
        numbers = re.findall(r'^\d+\.', content, re.MULTILINE)
        if numbers and len(numbers) > 1:
            # Check if numbers are sequential
            try:
                nums = [int(n.rstrip('.')) for n in numbers]
                is_sequential = all(nums[i] + 1 == nums[i + 1] for i in range(len(nums) - 1))
                if is_sequential and nums[0] == 1:
                    return True
            except:
                pass
        
        return has_introduction or has_conclusion
    
    def _select_chunking_strategy(self, group: List[DocumentSection], 
                                group_info: Dict, topics: List[str]) -> str:
        """Select the best chunking strategy for a group"""
        total_size = group_info['total_size']
        
        # If group fits comfortably in one chunk, keep together
        if total_size <= self.chunk_size:
            return 'keep_together'
        
        # If it's a complete semantic unit, try to keep together if possible
        if group_info['is_complete_unit'] and total_size <= self.max_chunk_size:
            return 'keep_together'
        
        # For procedural content, use sentence-based to preserve steps
        if group_info['type'] == 'procedural':
            return 'split_sentences'
        
        # For mixed content with clear topics, use semantic splitting
        if len(topics) > 1 and group_info['type'] == 'mixed':
            return 'split_semantic'
        
        # Default to sentence-based splitting
        return 'split_sentences'
    
    def _get_group_topics(self, sections: List[DocumentSection]) -> List[str]:
        """Get topics for a group of sections"""
        # Use semantic analyzer to get topics
        sections_with_topics = self.semantic_analyzer._assign_topics_to_sections(sections)
        
        all_topics = []
        for section_data in sections_with_topics:
            all_topics.extend(section_data['topics'])
        
        # Return unique topics, sorted by frequency
        from collections import Counter
        topic_counts = Counter(all_topics)
        return [topic for topic, _ in topic_counts.most_common()]
    
    def _format_group_optimally(self, group: List[DocumentSection], 
                              group_data: Dict) -> str:
        """Format a group of sections optimally based on content type"""
        parts = []
        
        # Add topic context if relevant
        if group_data.get('topics'):
            primary_topic = group_data['topics'][0]
            if primary_topic != 'general':
                parts.append(f"[Topic: {primary_topic.title()}]\n")
        
        # Format based on group type
        group_type = group_data.get('type', 'mixed')
        
        if group_type == 'procedural':
            # Emphasize structure for procedures
            for section in group:
                if section.title:
                    parts.append(f"\n## {section.title}\n")
                parts.append(section.content)
        else:
            # Standard formatting
            for section in group:
                if section.title:
                    parts.append(f"\n{section.title}\n")
                parts.append(section.content)
        
        return '\n'.join(parts).strip()
    
    def _split_by_sentences_with_context(self, group: List[DocumentSection], 
                                       group_data: Dict) -> List[Dict]:
        """Split by sentences while preserving context"""
        chunks = []
        
        # Merge all content
        merged_content = []
        for section in group:
            if section.title:
                merged_content.append(f"[{section.title}]")
            merged_content.append(section.content)
        
        full_text = '\n'.join(merged_content)
        sentences = sent_tokenize(full_text)
        
        # Create chunks with smart boundaries
        current_chunk = []
        current_size = 0
        
        # Add context header
        context_header = f"[Topic: {group_data['topics'][0].title()}]" if group_data.get('topics') else ""
        if context_header:
            current_chunk.append(context_header)
            current_size += len(context_header)
        
        for i, sentence in enumerate(sentences):
            sentence_size = len(sentence)
            
            # Check for natural breaking points
            is_natural_break = (
                i > 0 and (
                    sentences[i-1].rstrip().endswith('.') and 
                    sentence[0].isupper() and
                    not sentence.startswith(('However', 'Therefore', 'Additionally'))
                )
            )
            
            if current_size + sentence_size > self.chunk_size and current_chunk and is_natural_break:
                # Save chunk
                chunks.append({
                    'text': ' '.join(current_chunk),
                    'page': group[0].page_num,
                    'title': self._create_group_title(group)
                })
                
                # Start new chunk with context
                current_chunk = [context_header] if context_header else []
                current_chunk.append(sentence)
                current_size = len(context_header) + sentence_size
            else:
                current_chunk.append(sentence)
                current_size += sentence_size
        
        # Add final chunk
        if current_chunk:
            chunks.append({
                'text': ' '.join(current_chunk),
                'page': group[0].page_num,
                'title': self._create_group_title(group)
            })
        
        return chunks
    
    def _split_by_semantic_boundaries(self, group: List[DocumentSection], 
                                    group_data: Dict) -> List[Dict]:
        """Split at semantic boundaries"""
        # Delegate to semantic analyzer for topic-based splitting
        topics = group_data.get('topics', [])
        chunks = []
        
        # Group sections by subtopics
        current_chunk_sections = []
        current_size = 0
        current_topics = set()
        
        for section in group:
            section_topics = self._get_section_topics(section)
            section_size = len(section.content) + (len(section.title) if section.title else 0)
            
            # Check if this section introduces new topic
            new_topics = set(section_topics) - current_topics
            
            if new_topics and current_chunk_sections and current_size + section_size > self.chunk_size:
                # Save current chunk
                chunk_text = self._format_group_optimally(current_chunk_sections, group_data)
                chunks.append({
                    'text': chunk_text,
                    'page': current_chunk_sections[0].page_num,
                    'title': self._create_group_title(current_chunk_sections)
                })
                
                # Start new chunk
                current_chunk_sections = [section]
                current_size = section_size
                current_topics = set(section_topics)
            else:
                current_chunk_sections.append(section)
                current_size += section_size
                current_topics.update(section_topics)
        
        # Add final chunk
        if current_chunk_sections:
            chunk_text = self._format_group_optimally(current_chunk_sections, group_data)
            chunks.append({
                'text': chunk_text,
                'page': current_chunk_sections[0].page_num,
                'title': self._create_group_title(current_chunk_sections)
            })
        
        return chunks
    
    def _get_section_topics(self, section: DocumentSection) -> List[str]:
        """Get topics for a single section"""
        sections_with_topics = self.semantic_analyzer._assign_topics_to_sections([section])
        if sections_with_topics:
            return sections_with_topics[0]['topics']
        return []
    
    def _create_group_title(self, group: List[DocumentSection]) -> str:
        """Create an appropriate title for a group"""
        # Use first non-empty title
        for section in group:
            if section.title:
                if len(group) > 1:
                    return f"{section.title} (and related content)"
                return section.title
        
        # Fallback
        return "Related Information"
    
    def _are_structurally_related(self, section1: DocumentSection, 
                                section2: DocumentSection) -> bool:
        """Check structural relationship between sections"""
        # Check parent-child relationship
        if section2.level > section1.level:
            return True
        
        # Check sibling relationship
        if section1.level == section2.level:
            return self._are_sequential_sections(section1.title, section2.title)
        
        return False
    
    def _are_semantically_related(self, section1: DocumentSection, 
                                section2: DocumentSection) -> bool:
        """Check semantic relationship between sections"""
        return self._are_sections_related(section1, section2)
    
    def get_stats(self, chunks: List[Dict[str, str]]) -> Dict[str, any]:
        """Get statistics about the chunking results"""
        if not chunks:
            return {}
        
        chunk_lengths = [len(c['text']) for c in chunks]
        methods = [c.get('chunking_method', 'unknown') for c in chunks]
        chunk_types = [c.get('chunk_type', 'unknown') for c in chunks]
        
        return {
            "total_chunks": len(chunks),
            "avg_chunk_length": sum(chunk_lengths) / len(chunks),
            "min_chunk_length": min(chunk_lengths),
            "max_chunk_length": max(chunk_lengths),
            "chunking_methods_used": {
                method: methods.count(method) 
                for method in set(methods)
            },
            "chunk_types": {
                ctype: chunk_types.count(ctype) 
                for ctype in set(chunk_types)
            },
            "chunks_with_topics": sum(1 for c in chunks if c.get('topics'))
        }