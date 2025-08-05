import os
from typing import Dict, List, Set
from collections import defaultdict
import re

from nltk.tokenize import word_tokenize
from .base_chunk import BaseChunker, DocumentSection


class SemanticChunker(BaseChunker):
    """Chunks documents based on semantic similarity and topic coherence"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Aviation-specific keywords for topic detection
        self.aviation_topics = {
            'stall': ['stall', 'stalling', 'recovery', 'angle of attack', 'buffet', 'break'],
            'landing': ['landing', 'approach', 'touchdown', 'flare', 'ground effect', 'runway'],
            'takeoff': ['takeoff', 'departure', 'rotation', 'liftoff', 'climb', 'v1', 'vr'],
            'emergency': ['emergency', 'engine failure', 'fire', 'mayday', 'forced landing'],
            'navigation': ['navigation', 'vfr', 'ifr', 'waypoint', 'heading', 'course'],
            'weather': ['weather', 'wind', 'turbulence', 'visibility', 'cloud', 'icing'],
            'maneuvers': ['turn', 'bank', 'pitch', 'roll', 'chandelle', 'lazy eight'],
            'systems': ['engine', 'electrical', 'hydraulic', 'fuel', 'avionics', 'instrument'],
            'regulations': ['regulation', 'far', 'requirement', 'limitation', 'restriction'],
            'performance': ['performance', 'weight', 'balance', 'density altitude', 'speed']
        }
    
    def chunk_sections(self, sections: List[DocumentSection], source: str) -> List[Dict[str, str]]:
        """Chunk by semantic similarity (topic-based)"""
        chunks = []
        chunk_id = 0
        
        # First, assign topics to each section
        sections_with_topics = self._assign_topics_to_sections(sections)
        
        # Group sections by primary topic
        topic_groups = self._group_sections_by_topic(sections_with_topics)
        
        # Process each topic group
        for topic, topic_sections in topic_groups.items():
            # Further group by subtopics if needed
            subgroups = self._create_semantic_subgroups(topic_sections, topic)
            
            for subgroup in subgroups:
                # Calculate total size
                total_size = sum(
                    len(s['section'].content) + 
                    (len(s['section'].title) if s['section'].title else 0) 
                    for s in subgroup
                )
                
                if total_size <= self.max_chunk_size:
                    # Create single chunk for this semantic group
                    chunk_text = self._merge_semantic_sections(subgroup, topic)
                    chunks.append({
                        "text": chunk_text,
                        "page": str(subgroup[0]['section'].page_num),
                        "chunk_id": str(chunk_id),
                        "source": os.path.basename(source),
                        "title": self._create_semantic_title(subgroup, topic),
                        "chunking_method": "semantic",
                        "primary_topic": topic,
                        "topics": list(set(t for s in subgroup for t in s['topics'])),
                        "sections_count": len(subgroup)
                    })
                    chunk_id += 1
                else:
                    # Split large semantic group
                    split_chunks = self._split_semantic_group(subgroup, topic)
                    for chunk_data in split_chunks:
                        chunks.append({
                            "text": chunk_data['text'],
                            "page": str(chunk_data['page']),
                            "chunk_id": str(chunk_id),
                            "source": os.path.basename(source),
                            "title": chunk_data['title'],
                            "chunking_method": "semantic",
                            "primary_topic": topic,
                            "topics": chunk_data['topics'],
                            "sections_count": chunk_data['sections_count']
                        })
                        chunk_id += 1
        
        return chunks
    
    def _assign_topics_to_sections(self, sections: List[DocumentSection]) -> List[Dict]:
        """Assign topic labels to each section based on content"""
        sections_with_topics = []
        
        for section in sections:
            # Combine title and content for topic detection
            full_text = f"{section.title} {section.content}".lower()
            words = set(word_tokenize(full_text))
            
            # Find matching topics
            section_topics = []
            topic_scores = {}
            
            for topic, keywords in self.aviation_topics.items():
                score = sum(1 for keyword in keywords if keyword.lower() in full_text)
                if score > 0:
                    topic_scores[topic] = score
                    section_topics.append(topic)
            
            # If no specific topic found, assign 'general'
            if not section_topics:
                section_topics = ['general']
                topic_scores['general'] = 1
            
            # Determine primary topic (highest score)
            primary_topic = max(topic_scores.items(), key=lambda x: x[1])[0] if topic_scores else 'general'
            
            sections_with_topics.append({
                'section': section,
                'topics': section_topics,
                'primary_topic': primary_topic,
                'topic_scores': topic_scores
            })
        
        return sections_with_topics
    
    def _group_sections_by_topic(self, sections_with_topics: List[Dict]) -> Dict[str, List[Dict]]:
        """Group sections by their primary topic"""
        topic_groups = defaultdict(list)
        
        for section_data in sections_with_topics:
            topic_groups[section_data['primary_topic']].append(section_data)
        
        return dict(topic_groups)
    
    def _create_semantic_subgroups(self, topic_sections: List[Dict], topic: str) -> List[List[Dict]]:
        """Create subgroups within a topic based on semantic coherence"""
        if not topic_sections:
            return []
        
        subgroups = []
        current_subgroup = []
        current_size = 0
        
        for i, section_data in enumerate(topic_sections):
            section = section_data['section']
            section_size = len(section.content) + (len(section.title) if section.title else 0)
            
            # Check if this section should be in the current subgroup
            if current_subgroup:
                # Check semantic coherence with current subgroup
                if (self._is_semantically_coherent(current_subgroup, section_data) and 
                    current_size + section_size <= self.max_chunk_size):
                    current_subgroup.append(section_data)
                    current_size += section_size
                else:
                    # Start new subgroup
                    subgroups.append(current_subgroup)
                    current_subgroup = [section_data]
                    current_size = section_size
            else:
                # First section in subgroup
                current_subgroup.append(section_data)
                current_size = section_size
        
        # Add final subgroup
        if current_subgroup:
            subgroups.append(current_subgroup)
        
        return subgroups
    
    def _is_semantically_coherent(self, subgroup: List[Dict], new_section: Dict) -> bool:
        """Check if new section is semantically coherent with existing subgroup"""
        # Check topic overlap
        subgroup_topics = set()
        for s in subgroup:
            subgroup_topics.update(s['topics'])
        
        new_topics = set(new_section['topics'])
        
        # High coherence if significant topic overlap
        overlap = len(subgroup_topics & new_topics)
        if overlap >= 2 or (overlap >= 1 and len(new_topics) <= 2):
            return True
        
        # Check if sections are sequential or related
        last_section = subgroup[-1]['section']
        new_section_obj = new_section['section']
        
        if self._are_sections_related(last_section, new_section_obj):
            return True
        
        # Check title similarity
        if last_section.title and new_section_obj.title:
            if self._calculate_title_similarity(last_section.title, new_section_obj.title) > 0.5:
                return True
        
        return False
    
    def _calculate_title_similarity(self, title1: str, title2: str) -> float:
        """Calculate similarity between two titles"""
        words1 = set(word_tokenize(title1.lower()))
        words2 = set(word_tokenize(title2.lower()))
        
        # Remove stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of'}
        words1 = words1 - stop_words
        words2 = words2 - stop_words
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0
    
    def _merge_semantic_sections(self, sections: List[Dict], topic: str) -> str:
        """Merge sections with semantic coherence indicators"""
        parts = []
        
        # Add topic header
        parts.append(f"[Topic: {topic.title()}]\n")
        
        for section_data in sections:
            section = section_data['section']
            
            if section.title:
                parts.append(f"\n{section.title}")
            
            if section.content:
                parts.append(section.content)
            
            # Add semantic tags if multiple topics
            if len(section_data['topics']) > 1:
                other_topics = [t for t in section_data['topics'] if t != topic]
                if other_topics:
                    parts.append(f"\n[Also relates to: {', '.join(other_topics)}]")
        
        return '\n'.join(parts).strip()
    
    def _create_semantic_title(self, subgroup: List[Dict], topic: str) -> str:
        """Create a descriptive title for the semantic chunk"""
        # Use the first section's title if available
        if subgroup[0]['section'].title:
            base_title = subgroup[0]['section'].title
        else:
            base_title = f"{topic.title()} Information"
        
        # Add section count if multiple sections
        if len(subgroup) > 1:
            return f"{base_title} (and {len(subgroup)-1} related sections)"
        
        return base_title
    
    def _split_semantic_group(self, subgroup: List[Dict], topic: str) -> List[Dict]:
        """Split a large semantic group while maintaining coherence"""
        chunks = []
        current_sections = []
        current_size = 0
        
        for section_data in subgroup:
            section = section_data['section']
            section_size = len(section.content) + (len(section.title) if section.title else 0)
            
            if current_size + section_size > self.chunk_size and current_sections:
                # Create chunk
                chunk_text = self._merge_semantic_sections(current_sections, topic)
                chunks.append({
                    'text': chunk_text,
                    'page': current_sections[0]['section'].page_num,
                    'title': self._create_semantic_title(current_sections, topic),
                    'topics': list(set(t for s in current_sections for t in s['topics'])),
                    'sections_count': len(current_sections)
                })
                
                # Start new chunk with topic context
                current_sections = [section_data]
                current_size = section_size
            else:
                current_sections.append(section_data)
                current_size += section_size
        
        # Add final chunk
        if current_sections:
            chunk_text = self._merge_semantic_sections(current_sections, topic)
            chunks.append({
                'text': chunk_text,
                'page': current_sections[0]['section'].page_num,
                'title': self._create_semantic_title(current_sections, topic),
                'topics': list(set(t for s in current_sections for t in s['topics'])),
                'sections_count': len(current_sections)
            })
        
        return chunks
    
    def get_stats(self, chunks: List[Dict[str, str]]) -> Dict[str, any]:
        """Get statistics about the chunking results"""
        if not chunks:
            return {}
        
        chunk_lengths = [len(c['text']) for c in chunks]
        topics = [c.get('primary_topic', 'unknown') for c in chunks]
        all_topics = []
        for c in chunks:
            all_topics.extend(c.get('topics', []))
        
        return {
            "total_chunks": len(chunks),
            "avg_chunk_length": sum(chunk_lengths) / len(chunks),
            "min_chunk_length": min(chunk_lengths),
            "max_chunk_length": max(chunk_lengths),
            "unique_primary_topics": len(set(topics)),
            "chunks_by_topic": {
                topic: topics.count(topic) 
                for topic in set(topics)
            },
            "avg_topics_per_chunk": len(all_topics) / len(chunks) if chunks else 0,
            "most_common_topic": max(set(topics), key=topics.count) if topics else None
        }