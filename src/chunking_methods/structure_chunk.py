"""
Structure-based chunking strategy
Preserves document structure and keeps related sections together
"""
import os
from typing import Dict, List

from .base_chunk import BaseChunker, DocumentSection


class StructuralChunker(BaseChunker):
    """Chunks documents based on their structural hierarchy"""
    
    def chunk_sections(self, sections: List[DocumentSection], source: str) -> List[Dict[str, str]]:
        """Chunk by document structure (keep related headers and content together)"""
        chunks = []
        chunk_id = 0
        
        # Group sections by their hierarchical relationship
        section_groups = self._group_sections_hierarchically(sections)
        
        for group in section_groups:
            # Calculate total size of the group
            group_size = sum(len(s.content) + (len(s.title) if s.title else 0) for s in group)
            
            if group_size <= self.max_chunk_size:
                # Group fits in one chunk
                chunk_text = self._merge_sections(group)
                chunks.append({
                    "text": chunk_text,
                    "page": str(group[0].page_num),
                    "chunk_id": str(chunk_id),
                    "source": os.path.basename(source),
                    "title": group[0].title,
                    "chunking_method": "structural",
                    "sections_count": len(group),
                    "hierarchy_level": group[0].level
                })
                chunk_id += 1
            else:
                # Group too large, need to split
                sub_chunks = self._split_large_group(group)
                for sub_chunk in sub_chunks:
                    chunks.append({
                        "text": sub_chunk['text'],
                        "page": str(sub_chunk['page']),
                        "chunk_id": str(chunk_id),
                        "source": os.path.basename(source),
                        "title": sub_chunk['title'],
                        "chunking_method": "structural",
                        "sections_count": sub_chunk['sections_count'],
                        "hierarchy_level": sub_chunk['level']
                    })
                    chunk_id += 1
        
        return chunks
    
    def _group_sections_hierarchically(self, sections: List[DocumentSection]) -> List[List[DocumentSection]]:
        """Group sections based on their hierarchical relationship"""
        if not sections:
            return []
        
        groups = []
        current_group = []
        current_parent = None
        
        for section in sections:
            if not current_group:
                # Start first group
                current_group.append(section)
                current_parent = section
            else:
                # Check relationship with current group
                if self._is_child_section(current_parent, section):
                    # This section is a child of the current parent
                    current_group.append(section)
                elif self._is_sibling_section(current_parent, section):
                    # This section is a sibling - start new group
                    groups.append(current_group)
                    current_group = [section]
                    current_parent = section
                else:
                    # Higher level section - start new group
                    groups.append(current_group)
                    current_group = [section]
                    current_parent = section
        
        # Don't forget the last group
        if current_group:
            groups.append(current_group)
        
        return groups
    
    def _is_child_section(self, parent: DocumentSection, child: DocumentSection) -> bool:
        """Check if child section is hierarchically under parent"""
        # Level-based check
        if child.level > parent.level:
            return True
        
        # Number-based check (e.g., 1.1 is child of 1)
        if parent.title and child.title:
            parent_num = self._extract_section_number(parent.title)
            child_num = self._extract_section_number(child.title)
            
            if parent_num and child_num:
                return child_num.startswith(parent_num + '.')
        
        return False
    
    def _is_sibling_section(self, section1: DocumentSection, section2: DocumentSection) -> bool:
        """Check if sections are at the same hierarchical level"""
        # Same level check
        if section1.level == section2.level:
            # Additional check for numbered sections
            if section1.title and section2.title:
                num1 = self._extract_section_number(section1.title)
                num2 = self._extract_section_number(section2.title)
                
                if num1 and num2:
                    parts1 = num1.split('.')
                    parts2 = num2.split('.')
                    # Same depth in hierarchy
                    return len(parts1) == len(parts2)
            
            return True
        
        return False
    
    def _extract_section_number(self, title: str) -> str:
        """Extract section number from title"""
        import re
        match = re.match(r'^(\d+(?:\.\d+)*)', title)
        return match.group(1) if match else None
    
    def _split_large_group(self, group: List[DocumentSection]) -> List[Dict[str, any]]:
        """Split a large group of sections into smaller chunks"""
        chunks = []
        current_sections = []
        current_size = 0
        main_section = group[0]  # Parent section
        
        for section in group:
            section_size = len(section.content) + (len(section.title) if section.title else 0)
            
            if current_size + section_size > self.chunk_size and current_sections:
                # Save current chunk
                chunk_text = self._merge_sections(current_sections)
                chunks.append({
                    'text': chunk_text,
                    'page': current_sections[0].page_num,
                    'title': main_section.title,
                    'sections_count': len(current_sections),
                    'level': main_section.level
                })
                
                # Start new chunk
                # Include main section title for context if this is a subsection
                if section != main_section and main_section.title:
                    current_sections = [
                        DocumentSection(
                            title=f"{main_section.title} (continued)",
                            content="",
                            page_num=section.page_num,
                            section_type="header",
                            level=main_section.level
                        ),
                        section
                    ]
                    current_size = section_size + len(main_section.title) + 12  # "(continued)"
                else:
                    current_sections = [section]
                    current_size = section_size
            else:
                current_sections.append(section)
                current_size += section_size
        
        # Save remaining sections
        if current_sections:
            chunk_text = self._merge_sections(current_sections)
            chunks.append({
                'text': chunk_text,
                'page': current_sections[0].page_num,
                'title': main_section.title,
                'sections_count': len(current_sections),
                'level': main_section.level
            })
        
        return chunks
    
    def _merge_sections(self, sections: List[DocumentSection]) -> str:
        """Merge multiple sections into a single chunk preserving structure"""
        parts = []
        
        for i, section in enumerate(sections):
            if section.title:
                # Add appropriate formatting based on level
                if section.level == 1:
                    parts.append(f"\n# {section.title}\n")
                elif section.level == 2:
                    parts.append(f"\n## {section.title}\n")
                else:
                    parts.append(f"\n### {section.title}\n")
            
            if section.content:
                parts.append(section.content)
        
        return '\n'.join(parts).strip()
    
    def get_stats(self, chunks: List[Dict[str, str]]) -> Dict[str, any]:
        """Get statistics about the chunking results"""
        if not chunks:
            return {}
        
        section_counts = [c.get('sections_count', 1) for c in chunks]
        chunk_lengths = [len(c['text']) for c in chunks]
        hierarchy_levels = [c.get('hierarchy_level', 0) for c in chunks]
        
        return {
            "total_chunks": len(chunks),
            "avg_chunk_length": sum(chunk_lengths) / len(chunks),
            "min_chunk_length": min(chunk_lengths),
            "max_chunk_length": max(chunk_lengths),
            "avg_sections_per_chunk": sum(section_counts) / len(chunks),
            "unique_hierarchy_levels": len(set(hierarchy_levels)),
            "chunks_by_level": {
                level: hierarchy_levels.count(level) 
                for level in set(hierarchy_levels)
            }
        }