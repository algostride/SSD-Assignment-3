"""
VidyaVichar Code Similarity Analysis Framework
==============================================
A comprehensive framework for analyzing code similarity across MERN-stack projects
using textual, structural, and semantic similarity metrics.

Author: [Your Name]
Date: November 2025
"""

# ============================================================================
# PART 0: IMPORTS AND SETUP
# ============================================================================

import os
import re
import json
import hashlib
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from difflib import SequenceMatcher
import warnings
warnings.filterwarnings('ignore')

# Set visualization style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# ============================================================================
# PART A: PREPROCESSING & DATA UNDERSTANDING
# ============================================================================

class CodePreprocessor:
    """Handles preprocessing of MERN stack code files"""
    
    def __init__(self, base_dir):
        self.base_dir = Path(base_dir)
        self.valid_extensions = ['.js', '.jsx', '.json', '.css']
        
    def remove_comments(self, code, file_ext):
        """Remove single-line and multi-line comments"""
        if file_ext in ['.js', '.jsx']:
            # Remove single-line comments
            code = re.sub(r'//.*?$', '', code, flags=re.MULTILINE)
            # Remove multi-line comments
            code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
        elif file_ext == '.css':
            code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
        return code
    
    def normalize_code(self, code):
        """Normalize whitespace and formatting"""
        # Remove extra whitespace
        code = re.sub(r'\s+', ' ', code)
        # Remove leading/trailing whitespace
        code = code.strip()
        return code
    
    def is_minified(self, code, threshold=1000):
        """Check if code is minified (very long lines)"""
        lines = code.split('\n')
        avg_line_length = sum(len(line) for line in lines) / max(len(lines), 1)
        return avg_line_length > threshold
    
    def read_file(self, filepath):
        """Read file with error handling"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return f.read()
        except:
            try:
                with open(filepath, 'r', encoding='latin-1') as f:
                    return f.read()
            except:
                return ""
    
    def preprocess_file(self, filepath):
        """Complete preprocessing pipeline for a single file"""
        content = self.read_file(filepath)
        if not content:
            return ""
        
        ext = filepath.suffix
        
        # Skip minified files
        if self.is_minified(content):
            return ""
        
        # Remove comments
        content = self.remove_comments(content, ext)
        
        # Normalize formatting
        content = self.normalize_code(content)
        
        return content
    
    def analyze_project(self, project_path):
        """Analyze a single project and extract metrics"""
        project_path = Path(project_path)
        
        metrics = {
            'project_name': project_path.name,
            'total_files': 0,
            'js_files': 0,
            'jsx_files': 0,
            'json_files': 0,
            'css_files': 0,
            'total_loc': 0,
            'react_components': 0,
            'express_routes': 0,
            'mongoose_models': 0,
            'file_contents': {}
        }
        
        # Walk through project directory
        for root, dirs, files in os.walk(project_path):
            # Skip node_modules and build directories
            dirs[:] = [d for d in dirs if d not in ['node_modules', 'build', 'dist', '.git']]
            
            for file in files:
                filepath = Path(root) / file
                ext = filepath.suffix
                
                if ext in self.valid_extensions:
                    metrics['total_files'] += 1
                    
                    # Count by extension
                    if ext == '.js':
                        metrics['js_files'] += 1
                    elif ext == '.jsx':
                        metrics['jsx_files'] += 1
                    elif ext == '.json':
                        metrics['json_files'] += 1
                    elif ext == '.css':
                        metrics['css_files'] += 1
                    
                    # Preprocess and analyze content
                    content = self.preprocess_file(filepath)
                    if content:
                        # Count lines of code
                        # --- FIX: compute LOC from original file text (with comments removed)
                        # to preserve real line breaks rather than using the normalized text
                        raw = self.read_file(filepath)
                        if raw:
                            raw_no_comments = self.remove_comments(raw, ext)
                            loc = len([line for line in raw_no_comments.splitlines() if line.strip()])
                        else:
                            loc = 0
                        metrics['total_loc'] += loc
                        
                        # Detect React components
                        if ext in ['.js', '.jsx']:
                            if re.search(r'(class\s+\w+\s+extends\s+React\.Component|function\s+\w+\s*\([^)]*\)\s*{.*return.*<)', content):
                                metrics['react_components'] += 1
                            
                            # Detect Express routes
                            if re.search(r'(app|router)\.(get|post|put|delete|patch)', content):
                                metrics['express_routes'] += 1
                            
                            # Detect Mongoose models
                            if re.search(r'mongoose\.model\(|new\s+mongoose\.Schema', content):
                                metrics['mongoose_models'] += 1
                        
                        # Store relative path and content
                        rel_path = filepath.relative_to(project_path)
                        metrics['file_contents'][str(rel_path)] = content
        
        return metrics


class ProjectAnalyzer:
    """Analyzes multiple projects and generates summary statistics"""
    
    def __init__(self, projects_dir):
        self.projects_dir = Path(projects_dir)
        self.preprocessor = CodePreprocessor(projects_dir)
        self.projects_data = {}
    
    def analyze_all_projects(self):
        """Analyze all projects in the directory"""
        project_folders = [f for f in self.projects_dir.iterdir() if f.is_dir()]
        
        print(f"Found {len(project_folders)} projects to analyze...\n")
        
        for project_path in project_folders:
            print(f"Analyzing: {project_path.name}")
            metrics = self.preprocessor.analyze_project(project_path)
            self.projects_data[project_path.name] = metrics
            print(f"  - Files: {metrics['total_files']}, LOC: {metrics['total_loc']}")
        
        return self.projects_data
    
    def generate_summary_df(self):
        """Generate summary DataFrame"""
        summary_data = []
        
        for project_name, metrics in self.projects_data.items():
            summary_data.append({
                'Project': project_name,
                'Total Files': metrics['total_files'],
                'JS Files': metrics['js_files'],
                'JSX Files': metrics['jsx_files'],
                'JSON Files': metrics['json_files'],
                'CSS Files': metrics['css_files'],
                'Total LOC': metrics['total_loc'],
                'React Components': metrics['react_components'],
                'Express Routes': metrics['express_routes'],
                'Mongoose Models': metrics['mongoose_models']
            })
        
        return pd.DataFrame(summary_data)


# ============================================================================
# PART B: CODE SIMILARITY COMPUTATION
# ============================================================================

class TextualSimilarity:
    """Computes textual similarity using various metrics"""
    
    @staticmethod
    def levenshtein_similarity(s1, s2):
        """Compute similarity using SequenceMatcher (similar to Levenshtein)"""
        return SequenceMatcher(None, s1, s2).ratio()
    
    @staticmethod
    def tfidf_cosine_similarity(docs):
        """Compute TF-IDF based cosine similarity"""
        if len(docs) < 2:
            return np.array([[1.0]])
        
        vectorizer = TfidfVectorizer(
            analyzer='char',
            ngram_range=(3, 5),
            max_features=5000
        )
        
        try:
            tfidf_matrix = vectorizer.fit_transform(docs)
            similarity_matrix = cosine_similarity(tfidf_matrix)
            return similarity_matrix
        except:
            return np.eye(len(docs))
    
    @staticmethod
    def jaccard_similarity(s1, s2):
        """Compute Jaccard similarity between two strings"""
        set1 = set(s1.split())
        set2 = set(s2.split())
        
        if not set1 and not set2:
            return 1.0
        if not set1 or not set2:
            return 0.0
        
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        return intersection / union if union > 0 else 0.0


class StructuralSimilarity:
    """Computes structural similarity based on code structure"""
    
    @staticmethod
    def extract_structure(code):
        """Extract structural features from code"""
        features = {
            'functions': len(re.findall(r'function\s+\w+|const\s+\w+\s*=\s*\([^)]*\)\s*=>', code)),
            'classes': len(re.findall(r'class\s+\w+', code)),
            'imports': len(re.findall(r'import\s+.*from|require\(', code)),
            'exports': len(re.findall(r'export\s+(default|const|function|class)', code)),
            'async_functions': len(re.findall(r'async\s+function|async\s+\([^)]*\)\s*=>', code)),
            'try_catch': len(re.findall(r'try\s*{', code)),
            'conditionals': len(re.findall(r'if\s*\(', code)),
            'loops': len(re.findall(r'(for|while)\s*\(', code)),
        }
        return features
    
    @staticmethod
    def compare_structures(features1, features2):
        """Compare two feature dictionaries"""
        all_keys = set(features1.keys()).union(set(features2.keys()))
        
        differences = []
        for key in all_keys:
            val1 = features1.get(key, 0)
            val2 = features2.get(key, 0)
            max_val = max(val1, val2)
            if max_val > 0:
                differences.append(abs(val1 - val2) / max_val)
            else:
                differences.append(0)
        
        # Similarity is 1 - average difference
        similarity = 1 - (sum(differences) / len(differences)) if differences else 1.0
        return similarity
    
    @staticmethod
    def extract_api_routes(code):
        """Extract API routes from Express code"""
        routes = re.findall(r'(app|router)\.(get|post|put|delete|patch)\s*\(\s*[\'"]([^\'"]+)', code)
        return set(route[2] for route in routes)
    
    @staticmethod
    def extract_mongoose_schemas(code):
        """Extract Mongoose schema field names"""
        schema_blocks = re.findall(r'new\s+mongoose\.Schema\s*\(\s*{([^}]+)}', code, re.DOTALL)
        fields = set()
        for block in schema_blocks:
            field_names = re.findall(r'(\w+)\s*:', block)
            fields.update(field_names)
        return fields


class SemanticSimilarity:
    """Computes semantic similarity using embeddings"""
    
    @staticmethod
    def simple_code_embedding(code, embedding_dim=100):
        """Simple hash-based embedding for demonstration"""
        # In production, use CodeBERT or similar models
        # This is a simplified version for demonstration
        
        # Extract tokens
        tokens = re.findall(r'\w+', code.lower())
        
        # Create a simple embedding based on token frequency
        embedding = np.zeros(embedding_dim)
        
        for i, token in enumerate(set(tokens[:embedding_dim])):
            hash_val = int(hashlib.md5(token.encode()).hexdigest(), 16)
            embedding[hash_val % embedding_dim] += tokens.count(token)
        
        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        return embedding
    
    @staticmethod
    def compute_semantic_similarity(code1, code2):
        """Compute cosine similarity between code embeddings"""
        emb1 = SemanticSimilarity.simple_code_embedding(code1)
        emb2 = SemanticSimilarity.simple_code_embedding(code2)
        
        dot_product = np.dot(emb1, emb2)
        return max(0, min(1, dot_product))  # Clamp to [0, 1]


class SimilarityAnalyzer:
    """Main class for computing all similarity metrics"""
    
    def __init__(self, projects_data):
        self.projects_data = projects_data
        self.project_names = list(projects_data.keys())
        self.n_projects = len(self.project_names)
        
        # Initialize similarity matrices
        self.textual_similarity = np.zeros((self.n_projects, self.n_projects))
        self.structural_similarity = np.zeros((self.n_projects, self.n_projects))
        self.semantic_similarity = np.zeros((self.n_projects, self.n_projects))
        self.combined_similarity = np.zeros((self.n_projects, self.n_projects))
    
    def compute_all_similarities(self):
        """Compute all pairwise similarities"""
        print("\nComputing pairwise similarities...\n")
        
        for i, proj1 in enumerate(self.project_names):
            for j, proj2 in enumerate(self.project_names):
                if i == j:
                    # Same project = 100% similarity
                    self.textual_similarity[i][j] = 1.0
                    self.structural_similarity[i][j] = 1.0
                    self.semantic_similarity[i][j] = 1.0
                    self.combined_similarity[i][j] = 1.0
                elif i < j:  # Compute only upper triangle
                    print(f"Comparing {proj1} vs {proj2}")
                    
                    # Aggregate all file contents
                    content1 = " ".join(self.projects_data[proj1]['file_contents'].values())
                    content2 = " ".join(self.projects_data[proj2]['file_contents'].values())
                    
                    # Textual similarity
                    text_sim = TextualSimilarity.tfidf_cosine_similarity([content1, content2])[0][1]
                    self.textual_similarity[i][j] = text_sim
                    self.textual_similarity[j][i] = text_sim
                    
                    # Structural similarity
                    struct1 = StructuralSimilarity.extract_structure(content1)
                    struct2 = StructuralSimilarity.extract_structure(content2)
                    struct_sim = StructuralSimilarity.compare_structures(struct1, struct2)
                    self.structural_similarity[i][j] = struct_sim
                    self.structural_similarity[j][i] = struct_sim
                    
                    # Semantic similarity
                    sem_sim = SemanticSimilarity.compute_semantic_similarity(content1, content2)
                    self.semantic_similarity[i][j] = sem_sim
                    self.semantic_similarity[j][i] = sem_sim
                    
                    # Combined similarity (weighted average)
                    combined = 0.4 * text_sim + 0.3 * struct_sim + 0.3 * sem_sim
                    self.combined_similarity[i][j] = combined
                    self.combined_similarity[j][i] = combined
                    
                    print(f"  Textual: {text_sim:.3f}, Structural: {struct_sim:.3f}, Semantic: {sem_sim:.3f}")
        
        return {
            'textual': self.textual_similarity,
            'structural': self.structural_similarity,
            'semantic': self.semantic_similarity,
            'combined': self.combined_similarity
        }


# ============================================================================
# PART C: VISUALIZATION & REPORTING
# ============================================================================

class SimilarityVisualizer:
    """Creates visualizations for similarity analysis"""
    
    def __init__(self, project_names, similarity_matrices):
        self.project_names = project_names
        self.matrices = similarity_matrices
    
    def plot_heatmap(self, matrix_type='combined', save_path=None):
        """Plot similarity heatmap"""
        matrix = self.matrices[matrix_type]
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            matrix,
            annot=True,
            fmt='.3f',
            cmap='YlOrRd',
            xticklabels=self.project_names,
            yticklabels=self.project_names,
            vmin=0,
            vmax=1,
            cbar_kws={'label': 'Similarity Score'}
        )
        plt.title(f'{matrix_type.capitalize()} Similarity Heatmap', fontsize=16, fontweight='bold')
        plt.xlabel('Projects', fontsize=12)
        plt.ylabel('Projects', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_network_graph(self, matrix_type='combined', threshold=0.6, save_path=None):
        """Plot network graph of project similarities"""
        matrix = self.matrices[matrix_type]
        
        # Create graph
        G = nx.Graph()
        
        # Add nodes
        for name in self.project_names:
            G.add_node(name)
        
        # Add edges above threshold
        for i, proj1 in enumerate(self.project_names):
            for j, proj2 in enumerate(self.project_names):
                if i < j and matrix[i][j] > threshold:
                    G.add_edge(proj1, proj2, weight=matrix[i][j])
        
        # Draw graph
        plt.figure(figsize=(12, 10))
        pos = nx.spring_layout(G, k=2, iterations=50)
        
        # Draw nodes
        nx.draw_networkx_nodes(
            G, pos,
            node_color='lightblue',
            node_size=3000,
            alpha=0.9
        )
        
        # Draw edges with varying thickness
        edges = G.edges()
        weights = [G[u][v]['weight'] for u, v in edges]
        nx.draw_networkx_edges(
            G, pos,
            width=[w * 5 for w in weights],
            alpha=0.6,
            edge_color=weights,
            edge_cmap=plt.cm.Reds
        )
        
        # Draw labels
        nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')
        
        plt.title(f'Project Similarity Network (threshold={threshold})', fontsize=16, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_similarity_comparison(self, save_path=None):
        """Plot bar chart comparing average similarities by metric"""
        metrics = ['textual', 'structural', 'semantic', 'combined']
        avg_similarities = []
        
        for metric in metrics:
            matrix = self.matrices[metric]
            # Get upper triangle (excluding diagonal)
            upper_triangle = matrix[np.triu_indices_from(matrix, k=1)]
            avg_similarities.append(np.mean(upper_triangle))
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(metrics, avg_similarities, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
        plt.title('Average Similarity by Metric', fontsize=16, fontweight='bold')
        plt.xlabel('Similarity Metric', fontsize=12)
        plt.ylabel('Average Similarity Score', fontsize=12)
        plt.ylim(0, 1)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom', fontweight='bold')
        
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_distribution(self, matrix_type='combined', save_path=None):
        """Plot distribution of similarity scores"""
        matrix = self.matrices[matrix_type]
        upper_triangle = matrix[np.triu_indices_from(matrix, k=1)]
        
        plt.figure(figsize=(10, 6))
        plt.hist(upper_triangle, bins=20, color='#6C5CE7', alpha=0.7, edgecolor='black')
        plt.axvline(np.mean(upper_triangle), color='red', linestyle='--', 
                   linewidth=2, label=f'Mean: {np.mean(upper_triangle):.3f}')
        plt.axvline(np.median(upper_triangle), color='green', linestyle='--', 
                   linewidth=2, label=f'Median: {np.median(upper_triangle):.3f}')
        
        plt.title(f'{matrix_type.capitalize()} Similarity Distribution', fontsize=16, fontweight='bold')
        plt.xlabel('Similarity Score', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


# ============================================================================
# MAIN EXECUTION PIPELINE
# ============================================================================

def main(projects_directory='./projects'):
    """Main execution pipeline"""
    
    print("="*70)
    print("VidyaVichar Code Similarity Analysis")
    print("="*70)
    
    # Create results directory
    results_dir = Path('./results')
    results_dir.mkdir(exist_ok=True)
    
    # ========================================================================
    # PART A: Preprocessing & Data Understanding
    # ========================================================================
    print("\n" + "="*70)
    print("PART A: PREPROCESSING & DATA UNDERSTANDING")
    print("="*70)
    
    analyzer = ProjectAnalyzer(projects_directory)
    projects_data = analyzer.analyze_all_projects()
    
    # Generate summary
    summary_df = analyzer.generate_summary_df()
    print("\n" + summary_df.to_string(index=False))
    
    # Save summary
    summary_df.to_csv(results_dir / 'preprocessing_summary.csv', index=False)
    print(f"\n✓ Summary saved to {results_dir / 'preprocessing_summary.csv'}")
    
    # ========================================================================
    # PART B: Code Similarity Computation
    # ========================================================================
    print("\n" + "="*70)
    print("PART B: CODE SIMILARITY COMPUTATION")
    print("="*70)
    
    sim_analyzer = SimilarityAnalyzer(projects_data)
    similarity_matrices = sim_analyzer.compute_all_similarities()
    
    # Save similarity matrices
    for metric, matrix in similarity_matrices.items():
        matrix_df = pd.DataFrame(
            matrix,
            index=sim_analyzer.project_names,
            columns=sim_analyzer.project_names
        )
        matrix_df.to_csv(results_dir / f'{metric}_similarity_matrix.csv')
        print(f"✓ {metric.capitalize()} similarity matrix saved")
    
    # ========================================================================
    # PART C: Visualization & Reporting
    # ========================================================================
    print("\n" + "="*70)
    print("PART C: VISUALIZATION & REPORTING")
    print("="*70)
    
    visualizer = SimilarityVisualizer(sim_analyzer.project_names, similarity_matrices)
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    
    visualizer.plot_heatmap('combined', results_dir / 'heatmap_combined.png')
    visualizer.plot_heatmap('textual', results_dir / 'heatmap_textual.png')
    visualizer.plot_network_graph('combined', threshold=0.5, 
                                  save_path=results_dir / 'network_graph.png')
    visualizer.plot_similarity_comparison(results_dir / 'comparison_bar_chart.png')
    visualizer.plot_distribution('combined', results_dir / 'similarity_distribution.png')
    
    print("\n✓ All visualizations generated successfully!")
    
    # ========================================================================
    # Generate Insights Report
    # ========================================================================
    print("\n" + "="*70)
    print("KEY INSIGHTS")
    print("="*70)
    
    combined_matrix = similarity_matrices['combined']
    upper_triangle = combined_matrix[np.triu_indices_from(combined_matrix, k=1)]
    
    # Find most and least similar pairs
    max_idx = np.unravel_index(np.argmax(combined_matrix - np.eye(len(combined_matrix))), 
                                combined_matrix.shape)
    min_idx = np.unravel_index(np.argmin(combined_matrix + np.eye(len(combined_matrix)) * 2), 
                                combined_matrix.shape)
    
    print(f"\n📊 Overall Statistics:")
    print(f"  - Average Similarity: {np.mean(upper_triangle):.3f}")
    print(f"  - Median Similarity: {np.median(upper_triangle):.3f}")
    print(f"  - Std Deviation: {np.std(upper_triangle):.3f}")
    
    print(f"\n🔗 Most Similar Projects:")
    print(f"  - {sim_analyzer.project_names[max_idx[0]]} ↔ {sim_analyzer.project_names[max_idx[1]]}")
    print(f"  - Similarity Score: {combined_matrix[max_idx]:.3f}")
    
    print(f"\n🔀 Least Similar Projects:")
    print(f"  - {sim_analyzer.project_names[min_idx[0]]} ↔ {sim_analyzer.project_names[min_idx[1]]}")
    print(f"  - Similarity Score: {combined_matrix[min_idx]:.3f}")
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE!")
    print("="*70)
    print(f"\n📁 Results saved in: {results_dir.absolute()}")
    print("\nFiles generated:")
    print("  - preprocessing_summary.csv")
    print("  - *_similarity_matrix.csv (4 files)")
    print("  - *.png (5 visualization files)")
    
    return {
        'projects_data': projects_data,
        'summary': summary_df,
        'similarity_matrices': similarity_matrices,
        'visualizer': visualizer
    }


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    """
    To use this framework:
    
    1. Organize your projects in a directory structure like:
       projects/
       ├── team1_vidyavichar/
       ├── team2_vidyavichar/
       ├── team3_vidyavichar/
       └── ...
    
    2. Run the analysis:
       results = main('./projects')
    
    3. Access results:
       - results['summary']: DataFrame with project statistics
       - results['similarity_matrices']: Dict of similarity matrices
       - results['visualizer']: Visualizer object for custom plots
    
    4. Create custom visualizations:
       visualizer = results['visualizer']
       visualizer.plot_heatmap('textual')
       visualizer.plot_network_graph('structural', threshold=0.7)
    """
    
    # Example with sample data (replace with your actual projects path)
    # results = main('./path/to/your/projects')
    
    print("\n" + "="*70)
    print("FRAMEWORK READY!")
    print("="*70)
    
    results = main('./projects')