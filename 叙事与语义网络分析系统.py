# 叙事分析与语义网络分析系统
# Narrative Analysis & Semantic Network Analysis for Policy Documents
# 输出到Gephi格式 + 结构主题分析 + 主题偏向可视化

# 1. 导入必要的库
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from pathlib import Path
import logging
import time
import re
from typing import List, Dict, Tuple, Optional, Set
import warnings
warnings.filterwarnings('ignore')
import gc
from datetime import datetime
import json
from collections import Counter, defaultdict
import networkx as nx
from scipy.spatial.distance import cosine
try:
    import community as community_louvain
    LOUVAIN_AVAILABLE = True
except ImportError:
    try:
        import community_louvain
        LOUVAIN_AVAILABLE = True
    except ImportError:
        LOUVAIN_AVAILABLE = False
from scipy.stats import entropy
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo

# NLP相关导入
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.manifold import TSNE
import spacy

# 进度条 - 使用静默迭代器
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False


class SilentProgress:
    """静默进度迭代器 - 避免频繁刷新输出"""
    
    def __init__(self, iterable, desc: str = "", total: int = None, 
                 report_interval: int = 50, logger_func=None):
        self.iterable = iterable
        self.desc = desc
        self.total = total or len(iterable)
        self.report_interval = report_interval
        self.logger_func = logger_func or logger.info
        self.current = 0
        self.last_report = 0
    
    def __iter__(self):
        self.current = 0
        for item in self.iterable:
            self.current += 1
            # 每隔report_interval个或完成时报告一次
            if self.current - self.last_report >= self.report_interval or self.current == self.total:
                pct = (self.current / self.total) * 100
                self.logger_func(f"    └─ {self.desc}: {self.current}/{self.total} ({pct:.1f}%)")
                self.last_report = self.current
            yield item
    
    def __len__(self):
        return self.total


# ==================== 进度追踪器 ====================
class ProgressTracker:
    """进度追踪器 - 简洁的进度显示"""

    STAGES = [
        ("初始化配置", "⚙️"),
        ("加载停用词", "📚"),
        ("加载元数据", "📋"),
        ("加载文档", "📄"),
        ("叙事分析", "📖"),
        ("结构主题分析", "🎯"),
        ("语义网络分析", "🕸️"),
        ("主题偏向分析", "⚖️"),
        ("时间序列分析", "📈"),
        ("保存结果", "💾"),
        ("生成报告", "📊")
    ]

    def __init__(self):
        self.current_stage = 0
        self.stage_start_time = time.time()
        self.total_start_time = time.time()
        self.stage_times = {}
        self._last_progress_len = 0  # 用于覆盖输出

    def start_stage(self, stage_idx: int = None):
        """开始一个新阶段"""
        if stage_idx is not None:
            self.current_stage = stage_idx

        # 记录上一阶段耗时
        if self.current_stage > 0:
            elapsed = time.time() - self.stage_start_time
            prev_stage = self.STAGES[self.current_stage - 1][0]
            self.stage_times[prev_stage] = elapsed

        self.stage_start_time = time.time()

        if self.current_stage < len(self.STAGES):
            name, icon = self.STAGES[self.current_stage]
            progress = f"[{self.current_stage + 1}/{len(self.STAGES)}]"
            # 简洁的单行输出
            logger.info(f"\n{'─'*60}")
            logger.info(f"{icon} {progress} {name}")
            logger.info(f"{'─'*60}")

    def log_subtask(self, message: str, current: int = None, total: int = None):
        """记录子任务进度"""
        if current is not None and total is not None:
            pct = (current / total) * 100
            logger.info(f"  → {message} [{current}/{total}] ({pct:.1f}%)")
        else:
            logger.info(f"  → {message}")

    def log_metric(self, name: str, value):
        """记录指标"""
        logger.info(f"  ✓ {name}: {value}")

    def end_stage(self):
        """结束当前阶段"""
        self.current_stage += 1

    def finish(self):
        """完成所有阶段"""
        total_time = time.time() - self.total_start_time

        # 记录最后阶段耗时
        if self.current_stage > 0 and self.current_stage <= len(self.STAGES):
            elapsed = time.time() - self.stage_start_time
            prev_stage = self.STAGES[self.current_stage - 1][0]
            self.stage_times[prev_stage] = elapsed

        logger.info(f"\n{'═'*60}")
        logger.info("🎉 分析完成!")
        logger.info(f"{'═'*60}")
        logger.info(f"⏱️ 总耗时: {self._format_time(total_time)}")
        logger.info("")
        logger.info("📋 各阶段耗时:")
        for stage, t in self.stage_times.items():
            logger.info(f"  └─ {stage}: {self._format_time(t)}")
        logger.info("═" * 80)

    def _format_time(self, seconds: float) -> str:
        """格式化时间"""
        if seconds < 60:
            return f"{seconds:.1f}秒"
        elif seconds < 3600:
            mins = int(seconds // 60)
            secs = seconds % 60
            return f"{mins}分{secs:.0f}秒"
        else:
            hours = int(seconds // 3600)
            mins = int((seconds % 3600) // 60)
            return f"{hours}小时{mins}分"


# 全局进度追踪器
progress = ProgressTracker()

# 配置日志
log_file = Path(__file__).parent / 'narrative_analysis.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_file, encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)


# ==================== 配置类 ====================
class NarrativeAnalysisConfig:
    """叙事分析与语义网络分析配置
    
    所有配置项集中在此类中管理，便于统一修改和维护。
    使用方式：config = NarrativeAnalysisConfig()
    """
    
    def __init__(self):
        self._init_paths()
        self._init_filters()
        self._init_model_settings()
        self._init_text_processing()
        self._init_analysis_settings()
        self._init_visualization_settings()
        self._setup_output_dirs()
        
        logger.info(f"📁 输出目录: {self.output_dir}")
        if self.filter_countries:
            logger.info(f"🌍 国家筛选: {self.filter_countries}")
        if self.filter_org_types:
            logger.info(f"🏢 组织类型筛选: {self.filter_org_types}")
    
    def _init_paths(self):
        """初始化路径配置"""
        # ========== 数据路径配置 ==========
        # 元数据路径
        self.metadata_dir = Path("/Volumes/ZimingYe/A_project/12月数据采集汇总/数据标注/1226标注结果")
        
        # 数据源路径
        self.agora_fulltext_dir = Path("/Volumes/ZimingYe/A_project/12月数据采集汇总/translated_data/fulltext")
        self.original_data_dir = Path("/Volumes/ZimingYe/A_project/12月数据采集汇总/translated_data/DATA")
        
        # 输出目录
        self.base_output_dir = Path("/Volumes/ZimingYe/A_project/Agroa数据汇总分析/output")
        
        # 停用词路径
        self.stopwords_paths = [
            "/Users/ziming_ye/Downloads/stopwords-iso.json",
            "/Users/ziming_ye/Python/hit_stopwords.txt",
            "/Users/ziming_ye/Python/cn_all_stopwords.txt"
        ]
        
        # 模型路径
        self.embedding_model_path = '/Volumes/ZimingYe/Models'
    
    def _init_filters(self):
        """初始化数据筛选配置"""
        # ========== 数据筛选配置 ==========
        # 年份范围筛选
        self.min_valid_year = 2015
        self.max_valid_year = 2025
        
        # 国家筛选（为空列表表示不筛选，支持多个国家）
        # 可选值：'美国', '中国', '英国', '欧盟', '日本', '加拿大', '澳大利亚', '国际组织' 等
        # 示例：self.filter_countries = ['美国', '中国'] 表示只分析美国和中国的文档
        self.filter_countries = []  # 空列表表示不筛选，分析所有国家
        
        # 组织类型筛选（为空列表表示不筛选）
        # 可选值：'government', 'tech_company', 'international_org', 'academia', 'civil_society' 等
        self.filter_org_types = []  # 空列表表示不筛选
        
        # 文档类型筛选（为空列表表示不筛选）
        # 可选值：'national_guideline', 'white_paper', 'legislation', 'report', 'policy' 等
        self.filter_doc_types = []  # 空列表表示不筛选
        
        # 数据源筛选（为空列表表示不筛选）
        # 可选值：'fulltext', 'DATA'
        self.filter_data_sources = []  # 空列表表示不筛选
    
    def _init_model_settings(self):
        """初始化模型配置"""
        # ========== 模型配置 ==========
        self.spacy_model = 'en_core_web_sm'  # 或 'zh_core_web_sm' 中文
        self.n_topics = 15  # 主题数量
        self.topic_model_type = 'nmf'  # 'nmf' 或 'lda'
    
    def _init_text_processing(self):
        """初始化文本处理配置"""
        # ========== 文本处理配置 ==========
        self.max_doc_length = 5000000
        self.min_doc_length = 100
        self.min_word_freq = 3
    
    def _init_analysis_settings(self):
        """初始化分析参数配置"""
        # ========== 叙事分析配置 ==========
        self.narrative_window_size = 500  # 叙事分析滑动窗口大小
        self.narrative_segments = 10  # 将文档分割成多少段进行叙事分析
        
        # ========== 语义网络配置 ==========
        self.semantic_network_threshold = 0.3  # 语义关联阈值
        self.min_co_occurrence = 3  # 最小共现次数
        self.top_keywords_per_topic = 20  # 每个主题提取的关键词数
    
    def _init_visualization_settings(self):
        """初始化可视化配置"""
        # ========== 可视化配置 ==========
        self.visualization_dpi = 300
        self.color_palette = 'Set2'
    
    def _setup_output_dirs(self):
        """创建输出目录结构"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 如果有筛选条件，在目录名中体现
        filter_suffix = ""
        if self.filter_countries:
            filter_suffix += f"_countries_{'-'.join(self.filter_countries[:3])}"
            if len(self.filter_countries) > 3:
                filter_suffix += f"_{len(self.filter_countries)}"
        if self.filter_org_types:
            filter_suffix += f"_orgtypes_{len(self.filter_org_types)}"
        
        self.output_dir = self.base_output_dir / f"narrative_analysis_{timestamp}{filter_suffix}"
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Gephi输出目录
        self.gephi_output_dir = self.output_dir / "gephi"
        self.gephi_output_dir.mkdir(exist_ok=True)
        
        # 子目录
        for subdir in ['visualizations', 'reports', 'narrative', 'semantic_network', 'topic_bias', 'time_series']:
            (self.output_dir / subdir).mkdir(exist_ok=True)
    
    def get_filter_summary(self) -> Dict:
        """获取当前筛选条件的摘要"""
        filters = {}
        if self.filter_countries:
            filters['countries'] = self.filter_countries
        if self.filter_org_types:
            filters['org_types'] = self.filter_org_types
        if self.filter_doc_types:
            filters['doc_types'] = self.filter_doc_types
        if self.filter_data_sources:
            filters['data_sources'] = self.filter_data_sources
        filters['year_range'] = f"{self.min_valid_year}-{self.max_valid_year}"
        return filters
    
    def set_country_filter(self, countries: List[str]):
        """设置国家筛选
        
        Args:
            countries: 国家列表，如 ['美国', '中国'] 或 ['美国'] 分析单个国家
        """
        self.filter_countries = countries
        logger.info(f"🌍 已设置国家筛选: {countries}")
    
    def set_org_type_filter(self, org_types: List[str]):
        """设置组织类型筛选
        
        Args:
            org_types: 组织类型列表，如 ['government', 'tech_company']
        """
        self.filter_org_types = org_types
        logger.info(f"🏢 已设置组织类型筛选: {org_types}")
    
    def clear_filters(self):
        """清除所有筛选条件"""
        self.filter_countries = []
        self.filter_org_types = []
        self.filter_doc_types = []
        self.filter_data_sources = []
        logger.info("🧹 已清除所有筛选条件")


# ==================== 停用词加载 ====================
class StopwordsLoader:
    """停用词加载器"""
    
    def __init__(self, paths: List[str]):
        self.paths = paths
        self.stop_words = set()
    
    def load_all_stopwords(self) -> Set[str]:
        """加载所有停用词"""
        for path in self.paths:
            path = Path(path)
            if path.exists():
                if path.suffix == '.json':
                    with open(path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        if isinstance(data, dict):
                            for lang_words in data.values():
                                self.stop_words.update(lang_words)
                        elif isinstance(data, list):
                            self.stop_words.update(data)
                else:
                    with open(path, 'r', encoding='utf-8') as f:
                        self.stop_words.update(line.strip().lower() for line in f if line.strip())
        
        # 添加政策文档常见无意义词
        policy_stopwords = {
            'shall', 'may', 'must', 'should', 'would', 'could', 'can',
            'also', 'within', 'thereof', 'herein', 'therein', 'wherein',
            'pursuant', 'regarding', 'concerning', 'respectively',
            'article', 'section', 'chapter', 'paragraph', 'clause',
            'jktfrm', 'fmtsfmt', 'verdate', 'frmtd', 'verda', 'pfrm'
        }
        self.stop_words.update(policy_stopwords)
        
        # 添加报错词和数据提取错误产生的无意义词
        error_stopwords = {
            # 布尔值错误标记
            'false', 'true', 'null', 'none', 'nan',
            # 重复词（数据提取错误）
            'masked masked', 'data data', 'file file', 'page page',
            # 常见报错字符串
            'error', 'undefined', 'invalid', 'empty', 'missing',
            # 文件格式残留
            'utf', 'utf8', 'ascii', 'iso', 'bom'
        }
        self.stop_words.update(error_stopwords)
        
        # 添加所有数字（0-9组成的纯数字）
        # 生成0-99999的数字作为停用词
        for i in range(100000):
            self.stop_words.add(str(i))
        
        # 添加科学计数法和无意义数字组合
        numeric_stopwords = {
            # 科学计数法变体
            '1e', '2e', '3e', '4e', '5e', '6e', '7e', '8e', '9e', '0e',
            '1E', '2E', '3E', '4E', '5E', '6E', '7E', '8E', '9E', '0E',
            '1e2', '1e3', '1e4', '1e5', '1e6', '1e7', '1e8', '1e9',
            '2e2', '2e3', '2e4', '2e5', '2e6', '3e2', '3e3', '3e4', '3e5',
            # 零序列
            '00', '000', '0000', '00000', '000000', '0000000',
            # 数字字母混合无意义词
            'x000', 'x00', 'xx000', 'xxx000',
            '1a', '2a', '3a', '4a', '5a', '1b', '2b', '3b', '1c', '2c',
            'a1', 'a2', 'a3', 'b1', 'b2', 'b3', 'c1', 'c2', 'c3',
            # 罗马数字
            'i', 'ii', 'iii', 'iv', 'v', 'vi', 'vii', 'viii', 'ix', 'x',
            'xi', 'xii', 'xiii', 'xiv', 'xv',
            # 常见无意义组合
            '000 000', '00 00', '0000 0000',
        }
        self.stop_words.update(numeric_stopwords)
        
        logger.info(f"📚 已加载 {len(self.stop_words)} 个停用词（含数字和科学计数法）")
        return self.stop_words
    
    def is_stopword(self, word: str) -> bool:
        """检查是否为停用词"""
        word_lower = word.lower()
        
        # 检查是否在停用词表中
        if word_lower in self.stop_words:
            return True
        
        # 检查是否为纯数字
        if word_lower.isdigit():
            return True
        
        # 检查是否为数字+字母的无意义组合（如 "1e", "000x"）
        if re.match(r'^[0-9]+[a-zA-Z]?$', word_lower):
            return True
        if re.match(r'^[a-zA-Z]?[0-9]+$', word_lower):
            return True
        
        # 检查是否包含大量连续数字
        if re.match(r'.*[0-9]{3,}.*', word_lower):
            return True
        
        # 检查是否为科学计数法格式
        if re.match(r'^[0-9]+\.?[0-9]*[eE][+-]?[0-9]+$', word_lower):
            return True
        
        return False


# ==================== 元数据加载 ====================
class MetadataLoader:
    """元数据加载器 - 支持多字段筛选"""
    
    def __init__(self, metadata_dir: Path):
        self.metadata_dir = metadata_dir
        self.metadata_cache = {}
        self._load_metadata()
    
    def _load_metadata(self):
        """加载所有元数据文件"""
        # 过滤掉 macOS AppleDouble 元数据文件（以 ._ 开头）
        json_files = [f for f in self.metadata_dir.glob("*.json") if not f.name.startswith('._')]
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # 处理不同JSON格式
                    if isinstance(data, dict) and 'files' in data:
                        items = data['files']
                    elif isinstance(data, list):
                        items = data
                    else:
                        items = []
                    
                    for item in items:
                        if 'filename' in item:
                            # 去除.txt后缀（如果存在）
                            filename = item['filename']
                            if filename.endswith('.txt'):
                                filename = filename[:-4]
                            self.metadata_cache[filename] = item
            except Exception as e:
                logger.warning(f"⚠️ 加载元数据文件失败 {json_file}: {e}")
        
        logger.info(f"📚 已加载 {len(self.metadata_cache)} 条元数据")
    
    def get_metadata(self, filename: str) -> Dict:
        """获取文档元数据"""
        # 尝试多种文件名格式
        filename_clean = filename.replace('.txt', '')
        return self.metadata_cache.get(filename_clean, {})
    
    def get_year(self, filename: str, data_source: str) -> int:
        """获取文档年份"""
        metadata = self.get_metadata(filename)
        # 支持多种年份字段名
        year = metadata.get('year') or metadata.get('publication_year') or metadata.get('Year')
        return year if year else 2020
    
    def get_org_type(self, filename: str, data_source: str) -> str:
        """获取组织类型"""
        metadata = self.get_metadata(filename)
        # 支持多种字段名
        return metadata.get('level2_org_type') or metadata.get('org_type') or 'unknown'
    
    def get_country(self, filename: str, data_source: str) -> str:
        """获取国家/地区"""
        metadata = self.get_metadata(filename)
        # 支持多种字段名
        return metadata.get('level1_country_or_org') or metadata.get('country') or 'unknown'
    
    def get_doc_type(self, filename: str, data_source: str) -> str:
        """获取文档类型"""
        metadata = self.get_metadata(filename)
        return metadata.get('level3_doc_type') or metadata.get('doc_type') or 'unknown'
    
    def get_organization(self, filename: str, data_source: str) -> str:
        """获取组织名称"""
        metadata = self.get_metadata(filename)
        return metadata.get('organization') or metadata.get('Organization') or 'unknown'
    
    def get_tags(self, filename: str) -> List[str]:
        """获取标签列表"""
        metadata = self.get_metadata(filename)
        return metadata.get('tags', [])
    
    def get_all_countries(self) -> List[str]:
        """获取所有国家/地区列表"""
        countries = set()
        for metadata in self.metadata_cache.values():
            country = metadata.get('level1_country_or_org') or metadata.get('country')
            if country and country != 'unknown':
                countries.add(country)
        return sorted(list(countries))
    
    def get_all_org_types(self) -> List[str]:
        """获取所有组织类型列表"""
        org_types = set()
        for metadata in self.metadata_cache.values():
            org_type = metadata.get('level2_org_type') or metadata.get('org_type')
            if org_type and org_type != 'unknown':
                org_types.add(org_type)
        return sorted(list(org_types))
    
    def get_statistics(self) -> Dict:
        """获取元数据统计信息"""
        countries = Counter()
        org_types = Counter()
        years = Counter()
        
        for metadata in self.metadata_cache.values():
            country = metadata.get('level1_country_or_org') or metadata.get('country') or 'unknown'
            org_type = metadata.get('level2_org_type') or metadata.get('org_type') or 'unknown'
            year = metadata.get('year') or metadata.get('publication_year')
            
            countries[country] += 1
            org_types[org_type] += 1
            if year:
                years[year] += 1
        
        return {
            'total_documents': len(self.metadata_cache),
            'countries': dict(countries.most_common()),
            'org_types': dict(org_types.most_common()),
            'years': dict(sorted(years.items()))
        }


# ==================== 文档加载 ====================
class DocumentLoader:
    """文档加载器"""
    
    def __init__(self, agora_dir: Path, data_dir: Path, metadata_loader: MetadataLoader):
        self.agora_dir = agora_dir
        self.data_dir = data_dir
        self.metadata_loader = metadata_loader
    
    def load_documents(self) -> List[Dict]:
        """加载所有文档"""
        documents = []
        
        # 加载fulltext文件夹
        if self.agora_dir.exists():
            # 过滤掉 macOS AppleDouble 元数据文件（以 ._ 开头）
            for file_path in self.agora_dir.glob("*.txt"):
                if file_path.name.startswith('._'):
                    continue
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    documents.append({
                        'filename': file_path.stem,
                        'content': content,
                        'path': str(file_path),
                        'data_source': 'fulltext'
                    })
                except Exception as e:
                    logger.warning(f"⚠️ 加载文档失败 {file_path}: {e}")
        
        # 加载DATA文件夹
        if self.data_dir.exists():
            # 过滤掉 macOS AppleDouble 元数据文件（以 ._ 开头）
            for file_path in self.data_dir.glob("*.txt"):
                if file_path.name.startswith('._'):
                    continue
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    documents.append({
                        'filename': file_path.stem,
                        'content': content,
                        'path': str(file_path),
                        'data_source': 'DATA'
                    })
                except Exception as e:
                    logger.warning(f"⚠️ 加载文档失败 {file_path}: {e}")
        
        logger.info(f"📄 已加载 {len(documents)} 个文档")
        return documents


# ==================== 叙事分析器 ====================
class NarrativeAnalyzer:
    """叙事分析器 - 分析政策文本的叙事结构和语义演变"""
    
    def __init__(self, config: NarrativeAnalysisConfig):
        self.config = config
        self.stop_words = set()
        self.embedding_model = None
    
    def set_stop_words(self, stop_words: Set[str]):
        """设置停用词"""
        self.stop_words = stop_words
    
    def _is_numeric_word(self, word: str) -> bool:
        """检查是否为数字词（应被过滤）"""
        word_lower = word.lower()
        
        # 纯数字
        if word_lower.isdigit():
            return True
        
        # 科学计数法
        if re.match(r'^[0-9]+\.?[0-9]*[eE][+-]?[0-9]+$', word_lower):
            return True
        
        # 数字+e/数字+E组合
        if re.match(r'^[0-9]+[eE]$', word_lower):
            return True
        if re.match(r'^[eE][0-9]+$', word_lower):
            return True
        
        # 数字开头的词（如 "1e", "3e", "000x"）
        if re.match(r'^[0-9]+[a-zA-Z]*$', word_lower):
            return True
        
        # 包含3个以上连续数字
        if re.search(r'[0-9]{3,}', word_lower):
            return True
        
        return False
    
    def load_embedding_model(self):
        """加载嵌入模型"""
        if self.embedding_model is None:
            logger.info("  → 加载嵌入模型...")
            self.embedding_model = SentenceTransformer(self.config.embedding_model_path)
            logger.info("  ✓ 嵌入模型加载完成")
    
    def analyze_narrative_arc(self, documents: List[Dict]) -> Dict:
        """分析叙事弧 - 文档从头到尾的语义演变"""
        self.load_embedding_model()
        
        narrative_results = {
            'document_arcs': {},
            'global_patterns': {},
            'sentiment_flow': {}
        }
        
        all_segments = []
        
        for doc in SilentProgress(documents, desc="分析叙事弧", report_interval=100):
            content = doc['content']
            filename = doc['filename']
            
            # 分段
            segments = self._split_into_segments(content, self.config.narrative_segments)
            
            # 获取每段的嵌入
            segment_embeddings = []
            segment_texts = []
            
            for i, segment in enumerate(segments):
                if len(segment.strip()) > 50:
                    embedding = self.embedding_model.encode(segment)
                    segment_embeddings.append(embedding)
                    segment_texts.append({
                        'filename': filename,
                        'segment_idx': i,
                        'text': segment[:500]  # 保存前500字符
                    })
            
            if len(segment_embeddings) > 1:
                segment_embeddings = np.array(segment_embeddings)
                
                # 计算语义漂移
                semantic_shifts = []
                for i in range(1, len(segment_embeddings)):
                    shift = 1 - cosine_similarity(
                        segment_embeddings[i-1].reshape(1, -1),
                        segment_embeddings[i].reshape(1, -1)
                    )[0, 0]
                    semantic_shifts.append(float(shift))
                
                narrative_results['document_arcs'][filename] = {
                    'n_segments': len(segments),
                    'semantic_shifts': semantic_shifts,
                    'avg_shift': np.mean(semantic_shifts) if semantic_shifts else 0,
                    'max_shift': max(semantic_shifts) if semantic_shifts else 0,
                    'arc_type': self._classify_arc_type(semantic_shifts)
                }
                
                all_segments.extend(segment_texts)
        
        # 全局模式分析
        all_shifts = [arc['semantic_shifts'] for arc in narrative_results['document_arcs'].values()]
        if all_shifts:
            avg_arc = np.mean([np.mean(s) for s in all_shifts if s])
            narrative_results['global_patterns'] = {
                'avg_semantic_shift': avg_arc,
                'arc_distribution': Counter([arc['arc_type'] for arc in narrative_results['document_arcs'].values()])
            }
        
        logger.info(f"✅ 完成叙事弧分析: {len(narrative_results['document_arcs'])} 个文档")
        return narrative_results
    
    def _split_into_segments(self, text: str, n_segments: int) -> List[str]:
        """将文本分割成多个段"""
        # 按段落分割
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        if len(paragraphs) < n_segments:
            # 如果段落数不足，按句子分割
            sentences = re.split(r'[.!?。！？]', text)
            sentences = [s.strip() for s in sentences if s.strip()]
            paragraphs = sentences
        
        # 合并成指定数量的段
        segment_size = max(1, len(paragraphs) // n_segments)
        segments = []
        
        for i in range(0, len(paragraphs), segment_size):
            segment = ' '.join(paragraphs[i:i+segment_size])
            if segment.strip():
                segments.append(segment)
        
        return segments[:n_segments]
    
    def _classify_arc_type(self, shifts: List[float]) -> str:
        """分类叙事弧类型"""
        if not shifts:
            return 'unknown'
        
        avg_shift = np.mean(shifts)
        trend = shifts[-1] - shifts[0] if len(shifts) > 1 else 0
        
        if avg_shift < 0.1:
            return 'coherent'  # 连贯型
        elif trend > 0.1:
            return 'escalating'  # 递进型
        elif trend < -0.1:
            return 'concluding'  # 收束型
        else:
            return 'oscillating'  # 振荡型
    
    def analyze_discourse_patterns(self, documents: List[Dict]) -> Dict:
        """分析话语模式 - 关键概念的出现和演变"""
        
        # 提取所有关键词
        vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words=list(self.stop_words),
            ngram_range=(1, 2),
            min_df=3,
            token_pattern=r'(?u)\b(?![0-9]+\b)[a-zA-Z][a-zA-Z0-9]{2,}\b'  # 过滤纯数字
        )
        
        all_texts = [doc['content'] for doc in documents]
        tfidf_matrix = vectorizer.fit_transform(all_texts)
        feature_names = vectorizer.get_feature_names_out()
        
        # 分析每个文档的关键词分布
        discourse_patterns = {
            'top_terms': {},
            'term_evolution': {},
            'discourse_clusters': {}
        }
        
        # 全局top术语
        term_scores = np.array(tfidf_matrix.sum(axis=0)).flatten()
        top_indices = term_scores.argsort()[-50:][::-1]
        discourse_patterns['top_terms'] = {
            feature_names[i]: float(term_scores[i]) for i in top_indices
        }
        
        logger.info(f"  ✓ 话语模式: {len(discourse_patterns['top_terms'])} 个关键词")
        return discourse_patterns
    
    def create_narrative_visualizations(self, narrative_results: Dict, output_dir: Path):
        """创建叙事分析可视化"""
        files_created = []
        
        # 1. 叙事弧热力图
        arc_data = []
        for filename, arc in narrative_results['document_arcs'].items():
            for i, shift in enumerate(arc['semantic_shifts']):
                arc_data.append({
                    'document': filename[:30],  # 截断文件名
                    'segment': f"Seg {i+1}",
                    'shift': shift
                })
        
        if arc_data:
            df = pd.DataFrame(arc_data)
            pivot = df.pivot_table(index='document', columns='segment', values='shift', aggfunc='mean')
            
            fig = go.Figure(data=go.Heatmap(
                z=pivot.values,
                x=pivot.columns,
                y=pivot.index,
                colorscale='RdYlGn_r',
                hovertemplate='Document: %{y}<br>Segment: %{x}<br>Shift: %{z:.3f}<extra></extra>'
            ))
            
            fig.update_layout(
                title='Narrative Arc Semantic Shifts<br><sup>(叙事弧语义漂移热力图)</sup>',
                xaxis_title='Document Segment',
                yaxis_title='Document',
                height=max(600, len(pivot) * 20),
                paper_bgcolor='#F5FAFF',
                font=dict(color='#1565C0')
            )
            
            output_path = output_dir / 'narrative_arc_heatmap.html'
            fig.write_html(str(output_path))
            files_created.append('narrative_arc_heatmap.html')
        
        # 2. 弧类型分布饼图
        arc_types = narrative_results.get('global_patterns', {}).get('arc_distribution', {})
        if arc_types:
            fig2 = go.Figure(data=go.Pie(
                labels=list(arc_types.keys()),
                values=list(arc_types.values()),
                hole=0.4,
                marker_colors=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
            ))
            
            fig2.update_layout(
                title='Narrative Arc Types Distribution<br><sup>(叙事弧类型分布)</sup>',
                paper_bgcolor='#F5FAFF',
                font=dict(color='#1565C0')
            )
            
            output_path2 = output_dir / 'narrative_arc_types.html'
            fig2.write_html(str(output_path2))
            files_created.append('narrative_arc_types.html')
        
        if files_created:
            logger.info(f"  ✓ 叙事可视化: {', '.join(files_created)}")


# ==================== 语义网络分析器 ====================
class SemanticNetworkAnalyzer:
    """语义网络分析器 - 构建词语/概念网络并输出到Gephi"""
    
    def __init__(self, config: NarrativeAnalysisConfig):
        self.config = config
        self.stop_words = set()
    
    def set_stop_words(self, stop_words: Set[str]):
        """设置停用词"""
        self.stop_words = stop_words
    
    def _is_numeric_word(self, word: str) -> bool:
        """检查是否为数字词（应被过滤）"""
        word_lower = word.lower()
        
        # 纯数字
        if word_lower.isdigit():
            return True
        
        # 科学计数法
        if re.match(r'^[0-9]+\.?[0-9]*[eE][+-]?[0-9]+$', word_lower):
            return True
        
        # 数字+e/数字+E组合
        if re.match(r'^[0-9]+[eE]$', word_lower):
            return True
        if re.match(r'^[eE][0-9]+$', word_lower):
            return True
        
        # 数字开头的词（如 "1e", "3e", "000x"）
        if re.match(r'^[0-9]+[a-zA-Z]*$', word_lower):
            return True
        
        # 包含3个以上连续数字
        if re.search(r'[0-9]{3,}', word_lower):
            return True
        
        return False
    
    def build_semantic_network(self, documents: List[Dict]) -> nx.Graph:
        """构建语义网络"""
        G = nx.Graph()
        
        # 1. 提取关键词
        # 使用 token_pattern 过滤纯数字和短词
        vectorizer = TfidfVectorizer(
            max_features=500,
            stop_words=list(self.stop_words),
            ngram_range=(1, 2),
            min_df=3,
            token_pattern=r'(?u)\b(?![0-9]+\b)[a-zA-Z][a-zA-Z0-9]{2,}\b'  # 过滤纯数字，要求至少3个字符
        )
        
        all_texts = [doc['content'] for doc in documents]
        tfidf_matrix = vectorizer.fit_transform(all_texts)
        feature_names = vectorizer.get_feature_names_out()
        
        # 额外过滤：移除包含大量数字的词
        feature_names = [f for f in feature_names if not self._is_numeric_word(f)]
        
        logger.info(f"  ✓ 提取关键词: {len(feature_names)} 个")
        
        # 2. 计算词语共现
        co_occurrence = defaultdict(lambda: defaultdict(int))
        
        for doc in SilentProgress(documents, desc="计算共现", report_interval=100):
            content = doc['content'].lower()
            words_in_doc = set()
            
            for term in feature_names:
                if term in content:
                    words_in_doc.add(term)
            
            # 更新共现矩阵
            for w1 in words_in_doc:
                for w2 in words_in_doc:
                    if w1 < w2:  # 避免重复
                        co_occurrence[w1][w2] += 1
        
        # 3. 构建网络
        for w1, neighbors in co_occurrence.items():
            for w2, count in neighbors.items():
                if count >= self.config.min_co_occurrence:
                    weight = count / len(documents)  # 归一化
                    if weight >= self.config.semantic_network_threshold:
                        G.add_edge(w1, w2, weight=float(weight), co_occurrence=int(count))
        
        # 4. 添加节点属性（TF-IDF分数）
        term_scores = np.array(tfidf_matrix.sum(axis=0)).flatten()
        term_score_dict = {feature_names[i]: float(term_scores[i]) for i in range(len(feature_names))}
        
        for node in G.nodes():
            G.nodes[node]['tfidf_score'] = term_score_dict.get(node, 0)
            G.nodes[node]['degree'] = G.degree(node)
        
        logger.info(f"  ✓ 语义网络: {G.number_of_nodes()} 节点, {G.number_of_edges()} 边")
        return G
    
    def build_topic_network(self, documents: List[Dict], topic_results: Dict) -> nx.Graph:
        """构建主题网络"""
        
        G = nx.Graph()
        
        # 添加主题节点
        for topic_id, topic_data in topic_results.get('topics', {}).items():
            topic_name = topic_data.get('name', f'Topic {topic_id}')
            keywords = topic_data.get('keywords', [])
            
            G.add_node(
                f"Topic_{topic_id}",
                node_type='topic',
                label=topic_name,
                n_documents=topic_data.get('n_documents', 0),
                keywords=', '.join(keywords[:5])
            )
            
            # 添加关键词节点并连接
            for kw in keywords[:self.config.top_keywords_per_topic]:
                G.add_node(kw, node_type='keyword')
                G.add_edge(f"Topic_{topic_id}", kw, weight=1.0, edge_type='topic-keyword')
        
        # 计算主题间相似度并添加边
        topic_embeddings = topic_results.get('topic_embeddings', {})
        if topic_embeddings:
            topic_ids = list(topic_embeddings.keys())
            for i, t1 in enumerate(topic_ids):
                for t2 in topic_ids[i+1:]:
                    emb1 = np.array(topic_embeddings[t1])
                    emb2 = np.array(topic_embeddings[t2])
                    similarity = 1 - cosine(emb1, emb2)
                    
                    if similarity > 0.3:
                        G.add_edge(
                            f"Topic_{t1}", f"Topic_{t2}",
                            weight=float(similarity),
                            edge_type='topic-topic'
                        )
        
        logger.info(f"  ✓ 主题网络: {G.number_of_nodes()} 节点, {G.number_of_edges()} 边")
        return G
    
    def export_to_gephi(self, G: nx.Graph, output_path: Path, network_name: str):
        """导出到Gephi格式"""
        files_created = []
        
        # 导出为GEXF格式（Gephi原生格式）
        gexf_path = output_path / f"{network_name}.gexf"
        nx.write_gexf(G, str(gexf_path))
        files_created.append(f"{network_name}.gexf")
        
        # 导出为GraphML格式
        graphml_path = output_path / f"{network_name}.graphml"
        nx.write_graphml(G, str(graphml_path))
        files_created.append(f"{network_name}.graphml")
        
        # 导出节点和边CSV文件
        nodes_df = pd.DataFrame([
            {
                'Id': node,
                'Label': node,
                'node_type': G.nodes[node].get('node_type', 'unknown'),
                'tfidf_score': G.nodes[node].get('tfidf_score', 0),
                'degree': G.nodes[node].get('degree', 0)
            }
            for node in G.nodes()
        ])
        
        edges_df = pd.DataFrame([
            {
                'Source': u,
                'Target': v,
                'Weight': G.edges[u, v].get('weight', 1),
                'edge_type': G.edges[u, v].get('edge_type', 'unknown'),
                'co_occurrence': G.edges[u, v].get('co_occurrence', 0)
            }
            for u, v in G.edges()
        ])
        
        nodes_path = output_path / f"{network_name}_nodes.csv"
        edges_path = output_path / f"{network_name}_edges.csv"
        
        nodes_df.to_csv(nodes_path, index=False, encoding='utf-8-sig')
        edges_df.to_csv(edges_path, index=False, encoding='utf-8-sig')
        files_created.extend([f"{network_name}_nodes.csv", f"{network_name}_edges.csv"])
        
        # 计算网络统计指标
        stats = self._calculate_network_stats(G)
        stats_path = output_path / f"{network_name}_stats.json"
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        files_created.append(f"{network_name}_stats.json")
        
        logger.info(f"  ✓ Gephi导出({network_name}): {len(files_created)} 个文件")
        
        return stats
    
    def _calculate_network_stats(self, G: nx.Graph) -> Dict:
        """计算网络统计指标"""
        stats = {
            'basic': {
                'n_nodes': G.number_of_nodes(),
                'n_edges': G.number_of_edges(),
                'density': nx.density(G),
                'is_connected': nx.is_connected(G) if G.number_of_nodes() > 0 else False
            },
            'centrality': {}
        }
        
        if G.number_of_nodes() > 0:
            # 度中心性
            degree_centrality = nx.degree_centrality(G)
            stats['centrality']['degree'] = {
                'top_10': sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:10]
            }
            
            # 介数中心性（对于大网络可能较慢）
            if G.number_of_nodes() < 500:
                betweenness = nx.betweenness_centrality(G)
                stats['centrality']['betweenness'] = {
                    'top_10': sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:10]
                }
            
            # 聚类系数
            stats['clustering'] = {
                'avg_clustering': nx.average_clustering(G),
                'transitivity': nx.transitivity(G)
            }
            
            # 社区检测
            if G.number_of_edges() > 0:
                try:
                    communities = nx.community.greedy_modularity_communities(G)
                    stats['communities'] = {
                        'n_communities': len(communities),
                        'sizes': [len(c) for c in communities[:10]]
                    }
                except:
                    stats['communities'] = {'error': 'Community detection failed'}
        
        return stats
    
    def create_network_visualizations(self, G: nx.Graph, output_dir: Path, network_name: str):
        """创建网络可视化 - 支持多尺度子图和社区主题分类输出
        
        策略1: 多尺度子图 - 输出40/80/150节点的多个版本
        策略2: 基于社区的主题分类输出 - 使用Louvain算法检测社区，输出每个社区子图
        """
        if G.number_of_nodes() == 0:
            logger.warning(f"  ⚠️ 网络 {network_name} 为空，跳过可视化")
            return
        
        files_created = []
        
        try:
            # ========== 策略1: 多尺度子图输出 ==========
            logger.info(f"  📊 策略1: 多尺度子图输出...")
            
            # 定义目标尺度
            target_sizes = [40, 80, 150]
            degree_dict = dict(G.degree())
            
            for target_size in target_sizes:
                # 如果网络节点数不足，使用原网络
                if G.number_of_nodes() <= target_size:
                    G_viz = G
                else:
                    # 提取度中心性最高的节点
                    top_nodes = sorted(degree_dict.items(), key=lambda x: x[1], reverse=True)[:target_size]
                    top_node_names = [n for n, _ in top_nodes]
                    G_viz = G.subgraph(top_node_names).copy()
                
                # 生成可视化
                scale_name = f"{network_name}_{target_size}nodes"
                if self._generate_network_plotly(G_viz, output_dir, scale_name):
                    files_created.append(f"{scale_name}_interactive.html")
                    logger.info(f"  ✓ 多尺度子图({target_size}节点): {G_viz.number_of_nodes()} 节点, {G_viz.number_of_edges()} 边")
            
            # ========== 策略2: 基于社区的主题分类输出 ==========
            logger.info(f"  📊 策略2: 基于社区的主题分类输出...")
            
            if LOUVAIN_AVAILABLE and G.number_of_nodes() >= 20:
                try:
                    # 使用Louvain算法检测社区
                    communities = community_louvain.best_partition(G)
                    
                    # 统计社区信息
                    community_sizes = Counter(communities.values())
                    logger.info(f"  ✓ 检测到 {len(community_sizes)} 个社区")
                    
                    # 创建社区输出目录
                    community_dir = output_dir / f'{network_name}_communities'
                    community_dir.mkdir(exist_ok=True, parents=True)
                    
                    # 为每个社区生成子图
                    unique_communities = sorted(set(communities.values()))
                    
                    for comm_id in unique_communities:
                        comm_nodes = [n for n, c in communities.items() if c == comm_id]
                        
                        # 只处理节点数大于等于5的社区
                        if len(comm_nodes) >= 5:
                            G_comm = G.subgraph(comm_nodes).copy()
                            
                            # 生成社区可视化
                            comm_name = f"community_{comm_id}"
                            if self._generate_network_plotly(G_comm, community_dir, comm_name, 
                                                            title_suffix=f" (Community {comm_id}, {len(comm_nodes)} nodes)"):
                                files_created.append(f"communities/{comm_name}_interactive.html")
                                logger.info(f"  ✓ 社区{comm_id}: {G_comm.number_of_nodes()} 节点, {G_comm.number_of_edges()} 边")
                    
                    # 保存社区映射信息
                    community_map_path = output_dir / f'{network_name}_community_mapping.json'
                    # 将节点按社区分组
                    community_groups = defaultdict(list)
                    for node, comm_id in communities.items():
                        community_groups[int(comm_id)].append(node)
                    
                    with open(community_map_path, 'w', encoding='utf-8') as f:
                        json.dump({
                            'n_communities': len(community_sizes),
                            'community_sizes': {int(k): v for k, v in community_sizes.items()},
                            'community_groups': {str(k): v for k, v in community_groups.items()}
                        }, f, ensure_ascii=False, indent=2)
                    files_created.append(f"{network_name}_community_mapping.json")
                    logger.info(f"  ✓ 社区映射已保存: {network_name}_community_mapping.json")
                    
                except Exception as e:
                    logger.warning(f"  ⚠️ 社区检测失败: {e}，跳过策略2")
            else:
                if not LOUVAIN_AVAILABLE:
                    logger.warning("  ⚠️ community_louvain模块不可用，跳过策略2")
                else:
                    logger.info(f"  ⚠️ 网络节点数({G.number_of_nodes()})不足20，跳过社区检测")
            
            # 输出汇总
            logger.info(f"  ✅ 网络可视化完成: 共生成 {len(files_created)} 个文件")
            
        except Exception as e:
            logger.error(f"  ❌ 网络可视化失败: {e}")
            # 尝试保存网络统计信息作为备选
            try:
                stats_path = output_dir / f'{network_name}_stats.txt'
                with open(stats_path, 'w', encoding='utf-8') as f:
                    f.write(f"网络统计信息\n")
                    f.write(f"节点数: {G.number_of_nodes()}\n")
                    f.write(f"边数: {G.number_of_edges()}\n")
                    f.write(f"密度: {nx.density(G):.4f}\n")
                logger.info(f"  ✓ 网络统计已保存: {network_name}_stats.txt")
            except Exception as e2:
                logger.error(f"  ❌ 保存统计信息也失败: {e2}")
    
    def _generate_network_plotly(self, G: nx.Graph, output_dir: Path, network_name: str, 
                                  title_suffix: str = "") -> bool:
        """生成单个网络的Plotly交互式可视化
        
        Args:
            G: 网络图
            output_dir: 输出目录
            network_name: 网络名称
            title_suffix: 标题后缀
            
        Returns:
            bool: 是否成功生成
        """
        if G.number_of_nodes() == 0:
            return False
        
        try:
            # 计算布局
            if G.number_of_nodes() < 100:
                pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
            else:
                pos = nx.kamada_kawai_layout(G)
            
            # 提取边信息（限制边数以提高性能）
            max_edges = 2000
            edges = list(G.edges())
            if len(edges) > max_edges:
                edges_sorted = sorted(edges, key=lambda e: G.edges[e].get('weight', 0), reverse=True)
                edges = edges_sorted[:max_edges]
            
            edge_x = []
            edge_y = []
            for edge in edges:
                if edge[0] in pos and edge[1] in pos:
                    x0, y0 = pos[edge[0]]
                    x1, y1 = pos[edge[1]]
                    edge_x.extend([x0, x1, None])
                    edge_y.extend([y0, y1, None])
            
            # 边轨迹
            edge_trace = go.Scatter(
                x=edge_x, y=edge_y,
                line=dict(width=0.5, color='#888'),
                hoverinfo='none',
                mode='lines'
            )
            
            # 提取节点信息
            node_x = []
            node_y = []
            node_text = []
            node_color = []
            node_size = []
            
            for node in G.nodes():
                x, y = pos[node]
                node_x.append(x)
                node_y.append(y)
                
                degree = G.degree(node)
                tfidf = G.nodes[node].get('tfidf_score', 0)
                node_type = G.nodes[node].get('node_type', 'keyword')
                
                node_text.append(f"Node: {node}<br>Type: {node_type}<br>Degree: {degree}<br>TF-IDF: {tfidf:.3f}")
                node_size.append(min(20, 5 + degree * 2))
                node_color.append(degree)
            
            # 节点轨迹
            node_trace = go.Scatter(
                x=node_x, y=node_y,
                mode='markers',
                hoverinfo='text',
                text=node_text,
                marker=dict(
                    showscale=True,
                    colorscale='YlOrRd',
                    reversescale=True,
                    color=node_color,
                    size=node_size,
                    colorbar=dict(
                        thickness=15,
                        title=dict(text='Node Degree', side='right'),
                        xanchor='left'
                    ),
                    line_width=2
                )
            )
            
            fig = go.Figure(data=[edge_trace, node_trace],
                           layout=go.Layout(
                               title=f'Semantic Network: {network_name}{title_suffix}<br><sup>(语义网络可视化)</sup>',
                               titlefont_size=16,
                               showlegend=False,
                               hovermode='closest',
                               margin=dict(b=20, l=5, r=5, t=40),
                               annotations=[dict(
                                   text="Interactive network visualization",
                                   showarrow=False,
                                   xref="paper", yref="paper",
                                   x=0.005, y=-0.002
                               )],
                               xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                               yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                               paper_bgcolor='#F5FAFF',
                               font=dict(color='#1565C0')
                           ))
            
            output_path = output_dir / f'{network_name}_interactive.html'
            fig.write_html(str(output_path))
            return True
            
        except Exception as e:
            logger.warning(f"  ⚠️ 生成可视化 {network_name} 失败: {e}")
            return False


# ==================== 结构主题分析器 ====================
class StructuralTopicAnalyzer:
    """结构主题分析器 - 结合文档结构信息的主题模型"""
    
    def __init__(self, config: NarrativeAnalysisConfig):
        self.config = config
        self.stop_words = set()
        self.topic_model = None
        self.vectorizer = None
    
    def set_stop_words(self, stop_words: Set[str]):
        """设置停用词"""
        self.stop_words = stop_words
    
    def _is_numeric_word(self, word: str) -> bool:
        """检查是否为数字词（应被过滤）"""
        word_lower = word.lower()
        
        # 纯数字
        if word_lower.isdigit():
            return True
        
        # 科学计数法
        if re.match(r'^[0-9]+\.?[0-9]*[eE][+-]?[0-9]+$', word_lower):
            return True
        
        # 数字+e/数字+E组合
        if re.match(r'^[0-9]+[eE]$', word_lower):
            return True
        if re.match(r'^[eE][0-9]+$', word_lower):
            return True
        
        # 数字开头的词（如 "1e", "3e", "000x"）
        if re.match(r'^[0-9]+[a-zA-Z]*$', word_lower):
            return True
        
        # 包含3个以上连续数字
        if re.search(r'[0-9]{3,}', word_lower):
            return True
        
        return False
    
    def fit_topics(self, documents: List[Dict]) -> Dict:
        """训练主题模型"""
        logger.info(f"🎯 训练{self.config.topic_model_type.upper()}主题模型...")
        
        # 文本预处理
        all_texts = [doc['content'] for doc in documents]
        
        # 向量化 - 使用 token_pattern 过滤纯数字
        self.vectorizer = CountVectorizer(
            max_features=2000,
            stop_words=list(self.stop_words),
            ngram_range=(1, 2),
            min_df=3,
            max_df=0.9,
            token_pattern=r'(?u)\b(?![0-9]+\b)[a-zA-Z][a-zA-Z0-9]{2,}\b'  # 过滤纯数字
        )
        
        doc_term_matrix = self.vectorizer.fit_transform(all_texts)
        feature_names = self.vectorizer.get_feature_names_out()
        
        logger.info(f"  ✓ 文档-词矩阵: {doc_term_matrix.shape}")
        
        # 训练主题模型
        if self.config.topic_model_type == 'nmf':
            self.topic_model = NMF(
                n_components=self.config.n_topics,
                random_state=42,
                max_iter=200
            )
        else:  # LDA
            self.topic_model = LatentDirichletAllocation(
                n_components=self.config.n_topics,
                random_state=42,
                max_iter=20,
                learning_method='online'
            )
        
        doc_topic_matrix = self.topic_model.fit_transform(doc_term_matrix)
        topic_word_matrix = self.topic_model.components_
        
        # 提取主题关键词
        topics = {}
        for topic_idx in range(self.config.n_topics):
            top_word_indices = topic_word_matrix[topic_idx].argsort()[-30:][::-1]  # 取更多候选词
            keywords = [feature_names[i] for i in top_word_indices]
            
            # 过滤数字词和无意义词
            keywords = [kw for kw in keywords if not self._is_numeric_word(kw)][:15]  # 只保留前15个有效词
            
            # 生成主题名称（使用前两个有效关键词）
            if len(keywords) >= 2:
                topic_name = f"{keywords[0].title()} & {keywords[1].title()}"
            else:
                topic_name = f"Topic {topic_idx}"
            
            topics[topic_idx] = {
                'name': topic_name,
                'keywords': keywords,
                'weight': float(topic_word_matrix[topic_idx].sum())
            }
        
        # 计算文档主题分布
        doc_topic_dist = []
        for i, doc in enumerate(documents):
            topic_probs = doc_topic_matrix[i]
            dominant_topic = int(topic_probs.argmax())
            
            doc_topic_dist.append({
                'filename': doc['filename'],
                'dominant_topic': dominant_topic,
                'topic_distribution': {str(j): float(p) for j, p in enumerate(topic_probs)},
                'top_topics': sorted(enumerate(topic_probs), key=lambda x: x[1], reverse=True)[:3]
            })
        
        results = {
            'topics': topics,
            'doc_topic_distribution': doc_topic_dist,
            'n_topics': self.config.n_topics,
            'model_type': self.config.topic_model_type
        }
        
        logger.info(f"✅ 主题模型训练完成: {len(topics)} 个主题")
        return results
    
    def export_topic_wordlists(self, topic_results: Dict, output_dir: Path):
        """导出每个主题的Top20词表"""
        logger.info("📝 导出主题Top20词表...")
        
        topics = topic_results['topics']
        
        # 1. 导出TXT格式
        txt_path = output_dir / 'topic_top20_words.txt'
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("主题Top20关键词词表\n")
            f.write(f"模型类型: {topic_results['model_type'].upper()}\n")
            f.write(f"主题数量: {topic_results['n_topics']}\n")
            f.write("=" * 80 + "\n\n")
            
            for topic_idx in sorted(topics.keys()):
                topic = topics[topic_idx]
                f.write(f"【Topic {topic_idx}】 {topic['name']}\n")
                f.write("-" * 40 + "\n")
                f.write(f"Top 20 关键词:\n")
                for rank, word in enumerate(topic['keywords'][:20], 1):
                    f.write(f"  {rank:2d}. {word}\n")
                f.write(f"\n权重: {topic['weight']:.4f}\n")
                f.write("\n")
        
        logger.info(f"✅ TXT词表已保存: {txt_path}")
        
        # 2. 导出CSV格式（横向：每个主题一行）
        csv_data = []
        for topic_idx in sorted(topics.keys()):
            topic = topics[topic_idx]
            row = {
                'topic_id': topic_idx,
                'topic_name': topic['name'],
                'weight': topic['weight']
            }
            for i, word in enumerate(topic['keywords'][:20], 1):
                row[f'word_{i:02d}'] = word
            csv_data.append(row)
        
        csv_df = pd.DataFrame(csv_data)
        csv_path = output_dir / 'topic_top20_words.csv'
        csv_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        logger.info(f"✅ CSV词表已保存: {csv_path}")
        
        # 3. 导出详细CSV格式（纵向：每个词一行）
        detailed_data = []
        for topic_idx in sorted(topics.keys()):
            topic = topics[topic_idx]
            for rank, word in enumerate(topic['keywords'][:20], 1):
                detailed_data.append({
                    'topic_id': topic_idx,
                    'topic_name': topic['name'],
                    'rank': rank,
                    'keyword': word,
                    'topic_weight': topic['weight']
                })
        
        detailed_df = pd.DataFrame(detailed_data)
        detailed_path = output_dir / 'topic_top20_words_detailed.csv'
        detailed_df.to_csv(detailed_path, index=False, encoding='utf-8-sig')
        logger.info(f"✅ 详细CSV词表已保存: {detailed_path}")
        
        return {
            'txt_path': str(txt_path),
            'csv_path': str(csv_path),
            'detailed_csv_path': str(detailed_path)
        }
    
    def analyze_structural_effects(self, documents: List[Dict], topic_results: Dict, 
                                    metadata_loader: MetadataLoader) -> Dict:
        """分析结构效应 - 主题与元数据的关系"""
        logger.info("📐 分析结构效应...")
        
        # 按年份、组织类型、国家等分组分析主题分布
        year_topic_dist = defaultdict(lambda: defaultdict(float))
        org_topic_dist = defaultdict(lambda: defaultdict(float))
        
        for doc_topic in topic_results['doc_topic_distribution']:
            filename = doc_topic['filename']
            topic_dist = doc_topic['topic_distribution']
            
            year = metadata_loader.get_year(filename, 'fulltext')
            org_type = metadata_loader.get_org_type(filename, 'fulltext')
            
            for topic_id, prob in topic_dist.items():
                year_topic_dist[year][topic_id] += prob
                org_topic_dist[org_type][topic_id] += prob
        
        # 计算主题偏向
        results = {
            'year_topic_distribution': dict(year_topic_dist),
            'org_type_topic_distribution': dict(org_topic_dist),
            'topic_prevalence_over_time': {},
            'topic_prevalence_by_org': {}
        }
        
        # 计算每个主题随时间的变化趋势
        all_years = sorted(year_topic_dist.keys())
        for topic_id in range(self.config.n_topics):
            prevalence = [year_topic_dist[year].get(str(topic_id), 0) for year in all_years]
            results['topic_prevalence_over_time'][str(topic_id)] = {
                'years': all_years,
                'prevalence': prevalence
            }
        
        logger.info("✅ 结构效应分析完成")
        return results


# ==================== 时间序列分析器 ====================
class TimeSeriesAnalyzer:
    """时间序列分析器 - 分析政策文档的时间演变趋势"""
    
    def __init__(self, config: NarrativeAnalysisConfig):
        self.config = config
    
    def analyze_time_series(self, documents: List[Dict], metadata_loader: MetadataLoader,
                            topic_results: Dict = None) -> Dict:
        """综合时间序列分析"""
        logger.info("📈 开始时间序列分析...")
        
        results = {
            'document_timeline': {},
            'organization_timeline': {},
            'tag_timeline': {},
            'topic_timeline': {},
            'keyword_timeline': {},
            'statistics': {}
        }
        
        # 1. 文档时间线分析
        results['document_timeline'] = self._analyze_document_timeline(documents, metadata_loader)
        
        # 2. 组织活跃度时间线
        results['organization_timeline'] = self._analyze_organization_timeline(documents, metadata_loader)
        
        # 3. 标签时间演变
        results['tag_timeline'] = self._analyze_tag_timeline(documents, metadata_loader)
        
        # 4. 主题时间演变（如果有主题结果）
        if topic_results:
            results['topic_timeline'] = self._analyze_topic_timeline(documents, metadata_loader, topic_results)
        
        # 5. 关键词时间演变
        results['keyword_timeline'] = self._analyze_keyword_timeline(documents, metadata_loader)
        
        # 6. 统计摘要
        results['statistics'] = self._calculate_statistics(results)
        
        logger.info("✅ 时间序列分析完成")
        return results
    
    def _analyze_document_timeline(self, documents: List[Dict], metadata_loader: MetadataLoader) -> Dict:
        """分析文档时间线"""
        year_counts = Counter()
        year_by_source = defaultdict(Counter)
        year_by_org_type = defaultdict(Counter)
        
        for doc in documents:
            year = metadata_loader.get_year(doc['filename'], doc['data_source'])
            if self.config.min_valid_year <= year <= self.config.max_valid_year:
                year_counts[year] += 1
                year_by_source[doc['data_source']][year] += 1
                
                org_type = metadata_loader.get_org_type(doc['filename'], doc['data_source'])
                year_by_org_type[org_type][year] += 1
        
        years = sorted(year_counts.keys())
        
        return {
            'year_counts': dict(year_counts),
            'years': years,
            'counts': [year_counts[y] for y in years],
            'by_source': {k: dict(v) for k, v in year_by_source.items()},
            'by_org_type': {k: dict(v) for k, v in year_by_org_type.items()},
            'trend': self._calculate_trend(years, [year_counts[y] for y in years])
        }
    
    def _analyze_organization_timeline(self, documents: List[Dict], metadata_loader: MetadataLoader) -> Dict:
        """分析组织活跃度时间线"""
        org_year_docs = defaultdict(lambda: defaultdict(list))
        
        for doc in documents:
            year = metadata_loader.get_year(doc['filename'], doc['data_source'])
            if self.config.min_valid_year <= year <= self.config.max_valid_year:
                org_type = metadata_loader.get_org_type(doc['filename'], doc['data_source'])
                org_year_docs[org_type][year].append(doc['filename'])
        
        # 计算每个组织的活跃趋势
        org_trends = {}
        for org_type, year_docs in org_year_docs.items():
            years = sorted(year_docs.keys())
            counts = [len(year_docs[y]) for y in years]
            
            org_trends[org_type] = {
                'years': years,
                'counts': counts,
                'total_documents': sum(counts),
                'peak_year': years[counts.index(max(counts))] if counts else None,
                'trend': self._calculate_trend(years, counts)
            }
        
        # 按总文档数排序
        sorted_orgs = sorted(org_trends.items(), key=lambda x: x[1]['total_documents'], reverse=True)
        
        return {
            'organization_trends': dict(sorted_orgs),
            'top_organizations': [org for org, _ in sorted_orgs[:10]],
            'organization_count': len(org_trends)
        }
    
    def _analyze_tag_timeline(self, documents: List[Dict], metadata_loader: MetadataLoader) -> Dict:
        """分析标签时间演变"""
        tag_year_counts = defaultdict(lambda: defaultdict(int))
        
        for doc in documents:
            year = metadata_loader.get_year(doc['filename'], doc['data_source'])
            if self.config.min_valid_year <= year <= self.config.max_valid_year:
                metadata = metadata_loader.get_metadata(doc['filename'])
                tags = metadata.get('tags', [])
                
                for tag in tags:
                    tag_year_counts[tag][year] += 1
        
        # 计算每个标签的趋势
        all_years = sorted(set(year for tag_data in tag_year_counts.values() for year in tag_data.keys()))
        
        tag_trends = {}
        for tag, year_counts in tag_year_counts.items():
            if sum(year_counts.values()) >= 3:  # 只分析出现3次以上的标签
                counts = [year_counts.get(y, 0) for y in all_years]
                trend = self._calculate_trend(all_years, counts)
                
                tag_trends[tag] = {
                    'years': all_years,
                    'counts': counts,
                    'total': sum(counts),
                    'trend': trend,
                    'trend_direction': 'increasing' if trend > 0.05 else ('decreasing' if trend < -0.05 else 'stable'),
                    'peak_year': all_years[counts.index(max(counts))] if counts else None
                }
        
        # 分类标签
        increasing = sorted([(t, d) for t, d in tag_trends.items() if d['trend_direction'] == 'increasing'],
                          key=lambda x: x[1]['trend'], reverse=True)[:20]
        decreasing = sorted([(t, d) for t, d in tag_trends.items() if d['trend_direction'] == 'decreasing'],
                          key=lambda x: x[1]['trend'])[:20]
        
        return {
            'tag_trends': tag_trends,
            'all_years': all_years,
            'increasing_tags': [(t, d['trend']) for t, d in increasing],
            'decreasing_tags': [(t, d['trend']) for t, d in decreasing],
            'top_tags_by_count': sorted(tag_trends.items(), key=lambda x: x[1]['total'], reverse=True)[:30],
            'unique_tags': len(tag_trends)
        }
    
    def _analyze_topic_timeline(self, documents: List[Dict], metadata_loader: MetadataLoader,
                                topic_results: Dict) -> Dict:
        """分析主题时间演变"""
        # 获取文档主题分布
        doc_topic_dist = topic_results.get('doc_topic_distribution', [])
        
        # 建立文件名到主题的映射
        filename_to_topic = {}
        for doc_topic in doc_topic_dist:
            filename_to_topic[doc_topic['filename']] = doc_topic
        
        # 按年份统计主题分布
        topic_year_dist = defaultdict(lambda: defaultdict(float))
        year_doc_count = Counter()
        
        for doc in documents:
            year = metadata_loader.get_year(doc['filename'], doc['data_source'])
            if self.config.min_valid_year <= year <= self.config.max_valid_year:
                year_doc_count[year] += 1
                
                doc_topic = filename_to_topic.get(doc['filename'])
                if doc_topic:
                    for topic_id, prob in doc_topic.get('topic_distribution', {}).items():
                        topic_year_dist[topic_id][year] += prob
        
        all_years = sorted(year_doc_count.keys())
        topics = topic_results.get('topics', {})
        
        topic_trends = {}
        for topic_id in topics.keys():
            year_values = [topic_year_dist[str(topic_id)].get(y, 0) / max(year_doc_count[y], 1) 
                          for y in all_years]
            trend = self._calculate_trend(all_years, year_values)
            
            topic_trends[topic_id] = {
                'name': topics[topic_id].get('name', f'Topic {topic_id}'),
                'years': all_years,
                'values': year_values,
                'trend': trend,
                'trend_direction': 'increasing' if trend > 0.005 else ('decreasing' if trend < -0.005 else 'stable')
            }
        
        return {
            'topic_trends': topic_trends,
            'years': all_years,
            'increasing_topics': [tid for tid, td in topic_trends.items() if td['trend_direction'] == 'increasing'],
            'decreasing_topics': [tid for tid, td in topic_trends.items() if td['trend_direction'] == 'decreasing']
        }
    
    def _analyze_keyword_timeline(self, documents: List[Dict], metadata_loader: MetadataLoader) -> Dict:
        """分析关键词时间演变"""
        # 按年份提取关键词
        year_texts = defaultdict(list)
        
        for doc in documents:
            year = metadata_loader.get_year(doc['filename'], doc['data_source'])
            if self.config.min_valid_year <= year <= self.config.max_valid_year:
                year_texts[year].append(doc['content'])
        
        all_years = sorted(year_texts.keys())
        
        # 为每一年提取Top关键词
        year_keywords = {}
        keyword_year_freq = defaultdict(lambda: defaultdict(float))
        
        for year in all_years:
            texts = year_texts[year]
            combined_text = ' '.join(texts)
            
            # 使用TF提取关键词
            vectorizer = CountVectorizer(
                max_features=100,
                ngram_range=(1, 1),
                min_df=1,
                token_pattern=r'(?u)\b(?![0-9]+\b)[a-zA-Z][a-zA-Z0-9]{3,}\b'
            )
            
            try:
                tf_matrix = vectorizer.fit_transform([combined_text])
                feature_names = vectorizer.get_feature_names_out()
                tf_scores = tf_matrix.toarray()[0]
                
                # 获取Top关键词
                top_indices = tf_scores.argsort()[-30:][::-1]
                keywords = [(feature_names[i], tf_scores[i]) for i in top_indices]
                year_keywords[year] = keywords
                
                # 记录关键词频率
                for kw, score in keywords:
                    keyword_year_freq[kw][year] = score
            except:
                year_keywords[year] = []
        
        # 计算关键词趋势
        keyword_trends = {}
        for keyword, year_freq in keyword_year_freq.items():
            if sum(year_freq.values()) >= 2:  # 至少在2年出现
                values = [year_freq.get(y, 0) for y in all_years]
                trend = self._calculate_trend(all_years, values)
                
                keyword_trends[keyword] = {
                    'years': all_years,
                    'values': values,
                    'trend': trend,
                    'trend_direction': 'emerging' if trend > 0.1 else ('fading' if trend < -0.1 else 'stable')
                }
        
        # 找出新兴和消退关键词
        emerging = sorted([(k, d) for k, d in keyword_trends.items() if d['trend_direction'] == 'emerging'],
                         key=lambda x: x[1]['trend'], reverse=True)[:20]
        fading = sorted([(k, d) for k, d in keyword_trends.items() if d['trend_direction'] == 'fading'],
                       key=lambda x: x[1]['trend'])[:20]
        
        return {
            'year_keywords': year_keywords,
            'keyword_trends': keyword_trends,
            'emerging_keywords': [(k, d['trend']) for k, d in emerging],
            'fading_keywords': [(k, d['trend']) for k, d in fading],
            'years': all_years
        }
    
    def _calculate_trend(self, years: List[int], values: List[float]) -> float:
        """计算趋势系数（线性回归斜率）"""
        if len(years) < 2:
            return 0
        
        try:
            x = np.array(years)
            y = np.array(values)
            
            # 标准化以获得可比较的趋势值
            x_norm = (x - x.mean()) / (x.std() if x.std() > 0 else 1)
            y_norm = (y - y.mean()) / (y.std() if y.std() > 0 else 1)
            
            slope = np.polyfit(x_norm, y_norm, 1)[0]
            return float(slope)
        except:
            return 0
    
    def _calculate_statistics(self, results: Dict) -> Dict:
        """计算统计摘要"""
        doc_timeline = results.get('document_timeline', {})
        tag_timeline = results.get('tag_timeline', {})
        
        years = doc_timeline.get('years', [])
        counts = doc_timeline.get('counts', [])
        
        stats = {
            'year_range': f"{min(years)}-{max(years)}" if years else "N/A",
            'total_years': len(years),
            'total_documents': sum(counts),
            'avg_documents_per_year': np.mean(counts) if counts else 0,
            'peak_year': years[counts.index(max(counts))] if counts else None,
            'peak_documents': max(counts) if counts else 0,
            'unique_tags': tag_timeline.get('unique_tags', 0),
            'increasing_tags': len(tag_timeline.get('increasing_tags', [])),
            'decreasing_tags': len(tag_timeline.get('decreasing_tags', []))
        }
        
        return stats
    
    def create_time_series_visualizations(self, time_series_results: Dict, output_dir: Path):
        """创建时间序列可视化"""
        logger.info("📊 生成时间序列可视化...")
        
        files_created = []
        
        # 1. 文档数量时间线图
        doc_timeline = time_series_results.get('document_timeline', {})
        if doc_timeline.get('years'):
            fig = go.Figure()
            
            years = doc_timeline['years']
            counts = doc_timeline['counts']
            
            # 总文档趋势
            fig.add_trace(go.Scatter(
                x=years, y=counts,
                mode='lines+markers+text',
                name='Total Documents',
                line=dict(color='#1565C0', width=3),
                marker=dict(size=10),
                text=counts,
                textposition='top center',
                hovertemplate='Year: %{x}<br>Documents: %{y}<extra></extra>'
            ))
            
            # 按数据源添加趋势线
            by_source = doc_timeline.get('by_source', {})
            colors = ['#FF6B6B', '#4ECDC4']
            for i, (source, source_data) in enumerate(by_source.items()):
                source_counts = [source_data.get(y, 0) for y in years]
                fig.add_trace(go.Scatter(
                    x=years, y=source_counts,
                    mode='lines+markers',
                    name=source,
                    line=dict(color=colors[i % len(colors)], width=2, dash='dash'),
                    hovertemplate=f'{source}<br>Year: %{{x}}<br>Documents: %{{y}}<extra></extra>'
                ))
            
            fig.update_layout(
                title='Document Timeline Analysis<br><sup>(文档时间线分析)</sup>',
                xaxis_title='Year',
                yaxis_title='Number of Documents',
                height=500,
                hovermode='x unified',
                paper_bgcolor='#F5FAFF',
                font=dict(color='#1565C0'),
                legend=dict(orientation='h', yanchor='bottom', y=1.02)
            )
            
            output_path = output_dir / 'document_timeline.html'
            fig.write_html(str(output_path))
            files_created.append('document_timeline.html')
        
        # 2. 标签时间演变热力图
        tag_timeline = time_series_results.get('tag_timeline', {})
        tag_trends = tag_timeline.get('tag_trends', {})
        
        if tag_trends:
            # 选择Top30标签
            top_tags = sorted(tag_trends.items(), key=lambda x: x[1]['total'], reverse=True)[:30]
            years = tag_timeline.get('all_years', [])
            
            if top_tags and years:
                # 构建热力图矩阵
                tags = [t[0] for t in top_tags]
                matrix = []
                for tag, data in top_tags:
                    matrix.append(data['counts'])
                
                fig = go.Figure(data=go.Heatmap(
                    z=matrix,
                    x=years,
                    y=tags,
                    colorscale='Blues',
                    hovertemplate='Tag: %{y}<br>Year: %{x}<br>Count: %{z}<extra></extra>'
                ))
                
                fig.update_layout(
                    title='Tag Evolution Over Time<br><sup>(标签时间演变热力图)</sup>',
                    xaxis_title='Year',
                    yaxis_title='Tag',
                    height=max(600, len(tags) * 20),
                    paper_bgcolor='#F5FAFF',
                    font=dict(color='#1565C0')
                )
                
                output_path = output_dir / 'tag_evolution_heatmap.html'
                fig.write_html(str(output_path))
                files_created.append('tag_evolution_heatmap.html')
        
        # 3. 新兴/消退标签对比图
        if tag_timeline.get('increasing_tags') or tag_timeline.get('decreasing_tags'):
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=('Emerging Tags (新兴标签)', 'Fading Tags (消退标签)'),
                horizontal_spacing=0.15
            )
            
            # 新兴标签
            increasing = tag_timeline.get('increasing_tags', [])[:15]
            if increasing:
                fig.add_trace(go.Bar(
                    x=[t[1] for t in increasing],
                    y=[t[0] for t in increasing],
                    orientation='h',
                    marker_color='#4CAF50',
                    name='Emerging',
                    hovertemplate='Tag: %{y}<br>Trend: %{x:.3f}<extra></extra>'
                ), row=1, col=1)
            
            # 消退标签
            decreasing = tag_timeline.get('decreasing_tags', [])[:15]
            if decreasing:
                fig.add_trace(go.Bar(
                    x=[t[1] for t in decreasing],
                    y=[t[0] for t in decreasing],
                    orientation='h',
                    marker_color='#FF5722',
                    name='Fading',
                    hovertemplate='Tag: %{y}<br>Trend: %{x:.3f}<extra></extra>'
                ), row=1, col=2)
            
            fig.update_layout(
                title='Tag Trend Analysis<br><sup>(标签趋势分析)</sup>',
                height=max(500, max(len(increasing), len(decreasing)) * 25),
                showlegend=False,
                paper_bgcolor='#F5FAFF',
                font=dict(color='#1565C0')
            )
            
            output_path = output_dir / 'tag_trend_analysis.html'
            fig.write_html(str(output_path))
            files_created.append('tag_trend_analysis.html')
        
        # 4. 组织活跃度时间线
        org_timeline = time_series_results.get('organization_timeline', {})
        org_trends = org_timeline.get('organization_trends', {})
        
        if org_trends:
            top_orgs = org_timeline.get('top_organizations', [])[:10]
            
            fig = go.Figure()
            colors = px.colors.qualitative.Set2
            
            for i, org in enumerate(top_orgs):
                data = org_trends.get(org, {})
                years = data.get('years', [])
                counts = data.get('counts', [])
                
                if years and counts:
                    fig.add_trace(go.Scatter(
                        x=years, y=counts,
                        mode='lines+markers',
                        name=org[:30],
                        line=dict(color=colors[i % len(colors)], width=2),
                        hovertemplate=f'{org}<br>Year: %{{x}}<br>Documents: %{{y}}<extra></extra>'
                    ))
            
            fig.update_layout(
                title='Organization Activity Timeline<br><sup>(组织活跃度时间线)</sup>',
                xaxis_title='Year',
                yaxis_title='Number of Documents',
                height=600,
                hovermode='closest',
                paper_bgcolor='#F5FAFF',
                font=dict(color='#1565C0'),
                legend=dict(orientation='h', yanchor='bottom', y=-0.2)
            )
            
            output_path = output_dir / 'organization_timeline.html'
            fig.write_html(str(output_path))
            files_created.append('organization_timeline.html')
        
        # 5. 主题时间演变
        topic_timeline = time_series_results.get('topic_timeline', {})
        topic_trends = topic_timeline.get('topic_trends', {})
        
        if topic_trends:
            years = topic_timeline.get('years', [])
            
            fig = go.Figure()
            colors = px.colors.qualitative.Set2
            
            for i, (topic_id, data) in enumerate(sorted(topic_trends.items())[:15]):
                fig.add_trace(go.Scatter(
                    x=years, y=data.get('values', []),
                    mode='lines+markers',
                    name=f"T{topic_id}: {data.get('name', '')[:20]}",
                    line=dict(color=colors[i % len(colors)], width=2),
                    hovertemplate=f'Topic {topic_id}<br>Year: %{{x}}<br>Prevalence: %{{y:.3f}}<extra></extra>'
                ))
            
            fig.update_layout(
                title='Topic Evolution Over Time<br><sup>(主题时间演变)</sup>',
                xaxis_title='Year',
                yaxis_title='Topic Prevalence',
                height=600,
                hovermode='closest',
                paper_bgcolor='#F5FAFF',
                font=dict(color='#1565C0'),
                legend=dict(orientation='h', yanchor='bottom', y=-0.2)
            )
            
            output_path = output_dir / 'topic_evolution.html'
            fig.write_html(str(output_path))
            files_created.append('topic_evolution.html')
        
        # 6. 保存时间序列分析报告
        self._save_time_series_report(time_series_results, output_dir)
        files_created.append('time_series_report.txt')
        
        logger.info(f"  ✅ 时间序列可视化完成: {len(files_created)} 个文件")
        return files_created
    
    def _save_time_series_report(self, results: Dict, output_dir: Path):
        """保存时间序列分析报告"""
        report_path = output_dir / 'time_series_report.txt'
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("时间序列分析报告\n")
            f.write("Time Series Analysis Report\n")
            f.write("=" * 80 + "\n\n")
            
            # 统计摘要
            stats = results.get('statistics', {})
            f.write("📊 统计摘要\n")
            f.write("-" * 40 + "\n")
            f.write(f"  年份范围: {stats.get('year_range', 'N/A')}\n")
            f.write(f"  总年份数: {stats.get('total_years', 0)}\n")
            f.write(f"  总文档数: {stats.get('total_documents', 0)}\n")
            f.write(f"  年均文档数: {stats.get('avg_documents_per_year', 0):.1f}\n")
            f.write(f"  峰值年份: {stats.get('peak_year', 'N/A')} ({stats.get('peak_documents', 0)} 文档)\n")
            f.write(f"  唯一标签数: {stats.get('unique_tags', 0)}\n\n")
            
            # 文档时间线
            doc_timeline = results.get('document_timeline', {})
            f.write("📅 文档时间线\n")
            f.write("-" * 40 + "\n")
            years = doc_timeline.get('years', [])
            counts = doc_timeline.get('counts', [])
            for year, count in zip(years, counts):
                bar = '█' * min(int(count / max(counts) * 30) if counts else 0, 30)
                f.write(f"  {year}: {bar} ({count})\n")
            f.write("\n")
            
            # 新兴标签
            tag_timeline = results.get('tag_timeline', {})
            f.write("📈 新兴标签 (Top 15)\n")
            f.write("-" * 40 + "\n")
            for tag, trend in tag_timeline.get('increasing_tags', [])[:15]:
                f.write(f"  • {tag}: {trend:.3f}\n")
            f.write("\n")
            
            # 消退标签
            f.write("📉 消退标签 (Top 15)\n")
            f.write("-" * 40 + "\n")
            for tag, trend in tag_timeline.get('decreasing_tags', [])[:15]:
                f.write(f"  • {tag}: {trend:.3f}\n")
            f.write("\n")
            
            # 热门标签
            f.write("🔥 热门标签 (Top 20)\n")
            f.write("-" * 40 + "\n")
            for tag, data in tag_timeline.get('top_tags_by_count', [])[:20]:
                f.write(f"  • {tag}: {data.get('total', 0)} 次\n")
            f.write("\n")
            
            # 组织活跃度
            org_timeline = results.get('organization_timeline', {})
            f.write("🏢 活跃组织 (Top 15)\n")
            f.write("-" * 40 + "\n")
            for org, data in sorted(org_timeline.get('organization_trends', {}).items(),
                                   key=lambda x: x[1].get('total_documents', 0), reverse=True)[:15]:
                f.write(f"  • {org}: {data.get('total_documents', 0)} 文档, 峰值年份 {data.get('peak_year', 'N/A')}\n")
            f.write("\n")
            
            # 主题趋势
            topic_timeline = results.get('topic_timeline', {})
            if topic_timeline.get('topic_trends'):
                f.write("🎯 主题趋势分析\n")
                f.write("-" * 40 + "\n")
                f.write(f"  上升趋势主题: {len(topic_timeline.get('increasing_topics', []))} 个\n")
                f.write(f"  下降趋势主题: {len(topic_timeline.get('decreasing_topics', []))} 个\n\n")
            
            f.write("=" * 80 + "\n")
            f.write("报告生成完成\n")
        
        logger.info(f"  ✅ 时间序列报告已保存: {report_path}")


# ==================== 主题偏向分析器 ====================
class TopicBiasAnalyzer:
    """主题偏向分析器 - 分析不同群体的主题偏向"""
    
    def __init__(self, config: NarrativeAnalysisConfig):
        self.config = config
    
    def analyze_topic_bias(self, topic_results: Dict, structural_results: Dict,
                           metadata_loader: MetadataLoader, documents: List[Dict]) -> Dict:
        """分析主题偏向"""
        logger.info("⚖️ 开始主题偏向分析...")
        
        # 1. 按组织类型的主题偏向
        org_bias = self._calculate_group_bias(
            structural_results['org_type_topic_distribution'],
            topic_results['topics']
        )
        
        # 2. 按年份的主题偏向
        year_bias = self._calculate_temporal_bias(
            structural_results['year_topic_distribution'],
            topic_results['topics']
        )
        
        # 3. 主题显著性分析
        topic_prominence = self._calculate_topic_prominence(topic_results)
        
        results = {
            'organization_bias': org_bias,
            'temporal_bias': year_bias,
            'topic_prominence': topic_prominence
        }
        
        logger.info("✅ 主题偏向分析完成")
        return results
    
    def _calculate_group_bias(self, group_dist: Dict, topics: Dict) -> Dict:
        """计算群体偏向"""
        bias_results = {}
        
        # 计算全局分布
        global_dist = defaultdict(float)
        for group, topic_dist in group_dist.items():
            for topic_id, prob in topic_dist.items():
                global_dist[topic_id] += prob
        
        # 归一化
        total = sum(global_dist.values())
        if total > 0:
            global_dist = {k: v/total for k, v in global_dist.items()}
        
        # 计算每个群体的偏向
        for group, topic_dist in group_dist.items():
            total_group = sum(topic_dist.values())
            if total_group > 0:
                normalized = {k: v/total_group for k, v in topic_dist.items()}
                
                # 计算偏向分数（与全局分布的差异）
                bias_scores = {}
                for topic_id in topics.keys():
                    observed = normalized.get(str(topic_id), 0)
                    expected = global_dist.get(str(topic_id), 0)
                    if expected > 0:
                        bias_scores[topic_id] = (observed - expected) / expected
                    else:
                        bias_scores[topic_id] = 0
                
                # 找出最偏向的主题
                sorted_bias = sorted(bias_scores.items(), key=lambda x: abs(x[1]), reverse=True)
                bias_results[group] = {
                    'top_biased_topics': sorted_bias[:5],
                    'bias_direction': 'positive' if sum(bias_scores.values()) > 0 else 'negative'
                }
        
        return bias_results
    
    def _calculate_temporal_bias(self, year_dist: Dict, topics: Dict) -> Dict:
        """计算时间偏向"""
        years = sorted(year_dist.keys())
        
        if len(years) < 2:
            return {'error': 'Insufficient time points'}
        
        # 计算每个主题的时间趋势
        trends = {}
        for topic_id in topics.keys():
            values = [year_dist[year].get(str(topic_id), 0) for year in years]
            
            # 简单线性趋势
            x = np.arange(len(values))
            y = np.array(values)
            
            if len(x) > 1:
                slope = np.polyfit(x, y, 1)[0]
                trends[topic_id] = {
                    'slope': float(slope),
                    'direction': 'increasing' if slope > 0.01 else ('decreasing' if slope < -0.01 else 'stable'),
                    'values': values,
                    'years': years
                }
        
        # 分类主题
        increasing = [t for t, v in trends.items() if v['direction'] == 'increasing']
        decreasing = [t for t, v in trends.items() if v['direction'] == 'decreasing']
        stable = [t for t, v in trends.items() if v['direction'] == 'stable']
        
        return {
            'trends': trends,
            'increasing_topics': increasing,
            'decreasing_topics': decreasing,
            'stable_topics': stable
        }
    
    def _calculate_topic_prominence(self, topic_results: Dict) -> Dict:
        """计算主题显著性"""
        prominence = {}
        
        total_weight = sum(t['weight'] for t in topic_results['topics'].values())
        
        for topic_id, topic_data in topic_results['topics'].items():
            prominence[topic_id] = {
                'name': topic_data['name'],
                'relative_weight': topic_data['weight'] / total_weight if total_weight > 0 else 0,
                'keywords': topic_data['keywords'][:5]
            }
        
        return prominence
    
    def create_bias_visualizations(self, bias_results: Dict, output_dir: Path):
        """创建主题偏向可视化"""
        logger.info("📊 生成主题偏向可视化...")
        
        files_created = []
        
        # 1. 组织类型主题偏向热力图
        org_bias = bias_results.get('organization_bias', {})
        if org_bias:
            try:
                # 准备数据
                all_topics = set()
                for group_data in org_bias.values():
                    for topic_id, _ in group_data.get('top_biased_topics', []):
                        all_topics.add(topic_id)
                
                if all_topics:
                    topics = sorted(all_topics)
                    groups = list(org_bias.keys())
                    
                    matrix = []
                    for group in groups:
                        row = []
                        bias_dict = dict(org_bias[group].get('top_biased_topics', []))
                        for topic in topics:
                            row.append(bias_dict.get(topic, 0))
                        matrix.append(row)
                    
                    fig = go.Figure(data=go.Heatmap(
                        z=matrix,
                        x=[f"Topic {t}" for t in topics],
                        y=groups,
                        colorscale='RdYlGn',
                        zmid=0,
                        hovertemplate='Group: %{y}<br>Topic: %{x}<br>Bias: %{z:.3f}<extra></extra>'
                    ))
                    
                    fig.update_layout(
                        title='Topic Bias by Organization Type<br><sup>(组织类型主题偏向热力图)</sup>',
                        xaxis_title='Topic',
                        yaxis_title='Organization Type',
                        height=max(400, len(groups) * 40),
                        paper_bgcolor='#F5FAFF',
                        font=dict(color='#1565C0')
                    )
                    
                    output_path = output_dir / 'topic_bias_heatmap.html'
                    fig.write_html(str(output_path))
                    files_created.append('topic_bias_heatmap.html')
                    logger.info(f"✅ 主题偏向热力图已保存: {output_path}")
                else:
                    logger.warning("⚠️ 组织偏向数据中没有主题信息")
            except Exception as e:
                logger.error(f"❌ 生成组织偏向热力图失败: {e}")
        else:
            logger.warning("⚠️ 没有组织偏向数据，跳过热力图")
        
        # 2. 时间趋势折线图
        temporal_bias = bias_results.get('temporal_bias', {}).get('trends', {})
        if temporal_bias:
            try:
                fig = go.Figure()
                
                colors = px.colors.qualitative.Set2
                # 限制显示的主题数量
                max_topics = min(15, len(temporal_bias))
                for i, (topic_id, trend_data) in enumerate(list(temporal_bias.items())[:max_topics]):
                    fig.add_trace(go.Scatter(
                        x=trend_data['years'],
                        y=trend_data['values'],
                        name=f"Topic {topic_id}",
                        mode='lines+markers',
                        line=dict(color=colors[i % len(colors)]),
                        hovertemplate=f'Topic {topic_id}<br>Year: %{{x}}<br>Prevalence: %{{y:.3f}}<extra></extra>'
                    ))
                
                fig.update_layout(
                    title='Topic Prevalence Over Time<br><sup>(主题随时间变化趋势)</sup>',
                    xaxis_title='Year',
                    yaxis_title='Topic Prevalence',
                    height=600,
                    hovermode='closest',
                    paper_bgcolor='#F5FAFF',
                    font=dict(color='#1565C0'),
                    legend=dict(orientation='h', yanchor='bottom', y=-0.2)
                )
                
                output_path = output_dir / 'topic_trends.html'
                fig.write_html(str(output_path))
                files_created.append('topic_trends.html')
                logger.info(f"✅ 主题趋势图已保存: {output_path}")
            except Exception as e:
                logger.error(f"❌ 生成时间趋势图失败: {e}")
        else:
            logger.warning("⚠️ 没有时间趋势数据，跳过趋势图")
        
        # 3. 主题显著性条形图
        prominence = bias_results.get('topic_prominence', {})
        if prominence:
            try:
                topics_sorted = sorted(prominence.items(), key=lambda x: x[1]['relative_weight'], reverse=True)
                
                fig = go.Figure(go.Bar(
                    x=[f"T{t[0]}: {t[1]['name'][:20]}" for t in topics_sorted],
                    y=[t[1]['relative_weight'] for t in topics_sorted],
                    marker_color=['#FF6B6B' if t[1]['relative_weight'] > 0.1 else '#4ECDC4' for t in topics_sorted],
                    text=[f"{t[1]['relative_weight']:.1%}" for t in topics_sorted],
                    textposition='outside'
                ))
                
                fig.update_layout(
                    title='Topic Prominence<br><sup>(主题显著性分布)</sup>',
                    xaxis_title='Topic',
                    yaxis_title='Relative Weight',
                    height=600,
                    paper_bgcolor='#F5FAFF',
                    font=dict(color='#1565C0'),
                    xaxis_tickangle=-45
                )
                
                output_path = output_dir / 'topic_prominence.html'
                fig.write_html(str(output_path))
                files_created.append('topic_prominence.html')
                logger.info(f"✅ 主题显著性图已保存: {output_path}")
            except Exception as e:
                logger.error(f"❌ 生成主题显著性图失败: {e}")
        else:
            logger.warning("⚠️ 没有主题显著性数据，跳过显著性图")
        
        # 4. 保存偏向分析数据摘要（确保总有输出）
        try:
            summary_path = output_dir / 'bias_analysis_summary.txt'
            with open(summary_path, 'w', encoding='utf-8') as f:
                f.write("=" * 60 + "\n")
                f.write("主题偏向分析摘要\n")
                f.write("=" * 60 + "\n\n")
                
                # 组织偏向摘要
                f.write("组织偏向分析:\n")
                if org_bias:
                    for org, data in org_bias.items():
                        f.write(f"  - {org}: {len(data.get('top_biased_topics', []))} 个偏向主题\n")
                else:
                    f.write("  无数据\n")
                f.write("\n")
                
                # 时间趋势摘要
                f.write("时间趋势分析:\n")
                temporal = bias_results.get('temporal_bias', {})
                if isinstance(temporal, dict):
                    f.write(f"  上升趋势主题: {len(temporal.get('increasing_topics', []))} 个\n")
                    f.write(f"  下降趋势主题: {len(temporal.get('decreasing_topics', []))} 个\n")
                    f.write(f"  稳定趋势主题: {len(temporal.get('stable_topics', []))} 个\n")
                else:
                    f.write("  无数据\n")
                f.write("\n")
                
                # 显著性摘要
                f.write("主题显著性:\n")
                if prominence:
                    for tid, pdata in sorted(prominence.items(), key=lambda x: x[1]['relative_weight'], reverse=True)[:10]:
                        f.write(f"  Topic {tid}: {pdata['name'][:30]} ({pdata['relative_weight']:.1%})\n")
                else:
                    f.write("  无数据\n")
            
            files_created.append('bias_analysis_summary.txt')
            logger.info(f"✅ 偏向分析摘要已保存: {summary_path}")
        except Exception as e:
            logger.error(f"❌ 保存偏向分析摘要失败: {e}")
        
        logger.info(f"📊 主题偏向可视化完成: {len(files_created)} 个文件")


# ==================== 筛选辅助函数 ====================
def apply_filters(documents: List[Dict], metadata_loader: MetadataLoader, 
                  config: NarrativeAnalysisConfig) -> Tuple[List[Dict], Dict]:
    """应用数据筛选条件
    
    Args:
        documents: 文档列表
        metadata_loader: 元数据加载器
        config: 配置对象
    
    Returns:
        Tuple[List[Dict], Dict]: (筛选后的文档列表, 筛选统计信息)
    """
    filtered_docs = []
    stats = {
        'countries': Counter(),
        'org_types': Counter(),
        'years': Counter(),
        'sources': Counter(),
        'doc_types': Counter(),
        'filtered_by_country': 0,
        'filtered_by_org_type': 0,
        'filtered_by_year': 0,
        'filtered_by_source': 0
    }
    
    for doc in documents:
        filename = doc['filename']
        data_source = doc['data_source']
        
        # 获取元数据
        country = metadata_loader.get_country(filename, data_source)
        org_type = metadata_loader.get_org_type(filename, data_source)
        year = metadata_loader.get_year(filename, data_source)
        doc_type = metadata_loader.get_doc_type(filename, data_source)
        
        # 应用年份筛选
        if not (config.min_valid_year <= year <= config.max_valid_year):
            stats['filtered_by_year'] += 1
            continue
        
        # 应用国家筛选
        if config.filter_countries and country not in config.filter_countries:
            stats['filtered_by_country'] += 1
            continue
        
        # 应用组织类型筛选
        if config.filter_org_types and org_type not in config.filter_org_types:
            stats['filtered_by_org_type'] += 1
            continue
        
        # 应用文档类型筛选
        if config.filter_doc_types and doc_type not in config.filter_doc_types:
            continue
        
        # 应用数据源筛选
        if config.filter_data_sources and data_source not in config.filter_data_sources:
            stats['filtered_by_source'] += 1
            continue
        
        # 通过所有筛选条件
        filtered_docs.append(doc)
        
        # 更新统计
        stats['countries'][country] += 1
        stats['org_types'][org_type] += 1
        stats['years'][year] += 1
        stats['sources'][data_source] += 1
        stats['doc_types'][doc_type] += 1
    
    # 转换Counter为普通dict
    stats['countries'] = dict(stats['countries'])
    stats['org_types'] = dict(stats['org_types'])
    stats['years'] = dict(stats['years'])
    stats['sources'] = dict(stats['sources'])
    stats['doc_types'] = dict(stats['doc_types'])
    
    # 记录筛选日志
    if config.filter_countries:
        logger.info(f"🌍 国家筛选: {config.filter_countries} (排除 {stats['filtered_by_country']} 个文档)")
    if config.filter_org_types:
        logger.info(f"🏢 组织类型筛选: {config.filter_org_types} (排除 {stats['filtered_by_org_type']} 个文档)")
    if config.filter_data_sources:
        logger.info(f"📁 数据源筛选: {config.filter_data_sources} (排除 {stats['filtered_by_source']} 个文档)")
    
    return filtered_docs, stats


# ==================== 主分析流程 ====================
def run_full_analysis():
    """运行完整分析流程"""
    global progress
    
    logger.info("═" * 80)
    logger.info("🚀 叙事与语义网络分析系统启动")
    logger.info(f"📅 开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("═" * 80)
    
    # ========== 阶段1: 初始化配置 ==========
    progress.start_stage(0)
    config = NarrativeAnalysisConfig()
    progress.log_metric("输出目录", config.output_dir)
    progress.log_metric("主题数量", config.n_topics)
    progress.log_metric("主题模型", config.topic_model_type)
    progress.end_stage()
    
    # ========== 阶段2: 加载停用词 ==========
    progress.start_stage(1)
    stopwords_loader = StopwordsLoader(config.stopwords_paths)
    stop_words = stopwords_loader.load_all_stopwords()
    progress.log_metric("停用词数量", f"{len(stop_words)} 个")
    progress.end_stage()
    
    # ========== 阶段3: 加载元数据 ==========
    progress.start_stage(2)
    metadata_loader = MetadataLoader(config.metadata_dir)
    progress.log_metric("元数据条目", f"{len(metadata_loader.metadata_cache)} 条")
    progress.end_stage()
    
    # ========== 阶段4: 加载文档 ==========
    progress.start_stage(3)
    doc_loader = DocumentLoader(
        config.agora_fulltext_dir,
        config.original_data_dir,
        metadata_loader
    )
    documents = doc_loader.load_documents()
    
    if not documents:
        logger.error("❌ 没有找到有效文档")
        return
    
    # 应用筛选条件
    filtered_docs, filter_stats = apply_filters(documents, metadata_loader, config)
    
    progress.log_metric("原始文档数", f"{len(documents)} 个")
    progress.log_metric("过滤后文档数", f"{len(filtered_docs)} 个")
    progress.log_metric("年份范围", f"{config.min_valid_year}-{config.max_valid_year}")
    
    # 显示筛选统计
    if filter_stats['countries']:
        progress.log_metric("国家分布", dict(filter_stats['countries']))
    if filter_stats['org_types']:
        progress.log_metric("组织类型分布", dict(filter_stats['org_types']))
    if filter_stats['years']:
        progress.log_metric("年份分布", dict(filter_stats['years']))
    progress.log_metric("数据源分布", dict(filter_stats['sources']))
    
    if len(filtered_docs) == 0:
        logger.error("❌ 筛选后没有符合条件的文档")
        logger.info(f"💡 可用国家: {metadata_loader.get_all_countries()}")
        logger.info(f"💡 可用组织类型: {metadata_loader.get_all_org_types()}")
        return
    
    progress.end_stage()
    
    # ========== 阶段5: 叙事分析 ==========
    progress.start_stage(4)
    narrative_analyzer = NarrativeAnalyzer(config)
    narrative_analyzer.set_stop_words(stop_words)
    
    progress.log_subtask("分析叙事弧...")
    narrative_results = narrative_analyzer.analyze_narrative_arc(filtered_docs)
    
    progress.log_subtask("分析话语模式...")
    discourse_results = narrative_analyzer.analyze_discourse_patterns(filtered_docs)
    
    progress.log_subtask("生成叙事可视化...")
    narrative_analyzer.create_narrative_visualizations(narrative_results, config.output_dir / 'narrative')
    
    # 叙事分析统计
    arc_types = Counter([arc['arc_type'] for arc in narrative_results['document_arcs'].values()])
    progress.log_metric("叙事弧类型分布", dict(arc_types))
    progress.end_stage()
    
    # ========== 阶段6: 结构主题分析 ==========
    progress.start_stage(5)
    topic_analyzer = StructuralTopicAnalyzer(config)
    topic_analyzer.set_stop_words(stop_words)
    
    progress.log_subtask("拟合主题模型...")
    topic_results = topic_analyzer.fit_topics(filtered_docs)
    
    progress.log_subtask("分析结构效应...")
    structural_results = topic_analyzer.analyze_structural_effects(filtered_docs, topic_results, metadata_loader)
    
    progress.log_subtask("导出主题词表...")
    topic_analyzer.export_topic_wordlists(topic_results, config.output_dir / 'reports')
    
    # 主题分析统计
    n_topics = len(topic_results['topics'])
    progress.log_metric("提取主题数", n_topics)
    for tid, tdata in sorted(topic_results['topics'].items())[:5]:
        progress.log_metric(f"  Topic {tid}", tdata.get('name', 'Unknown')[:30])
    progress.end_stage()
    
    # ========== 阶段7: 语义网络分析 ==========
    progress.start_stage(6)
    network_analyzer = SemanticNetworkAnalyzer(config)
    network_analyzer.set_stop_words(stop_words)
    
    progress.log_subtask("构建语义网络...")
    semantic_network = network_analyzer.build_semantic_network(filtered_docs)
    network_stats = network_analyzer.export_to_gephi(
        semantic_network, 
        config.gephi_output_dir, 
        'semantic_network'
    )
    network_analyzer.create_network_visualizations(
        semantic_network,
        config.output_dir / 'semantic_network',
        'semantic_network'
    )
    
    progress.log_subtask("构建主题网络...")
    topic_network = network_analyzer.build_topic_network(filtered_docs, topic_results)
    network_analyzer.export_to_gephi(
        topic_network,
        config.gephi_output_dir,
        'topic_network'
    )
    network_analyzer.create_network_visualizations(
        topic_network,
        config.output_dir / 'semantic_network',
        'topic_network'
    )
    
    # 网络统计
    if network_stats and 'basic' in network_stats:
        progress.log_metric("网络节点数", network_stats['basic'].get('n_nodes', 0))
        progress.log_metric("网络边数", network_stats['basic'].get('n_edges', 0))
        progress.log_metric("网络密度", f"{network_stats['basic'].get('density', 0):.4f}")
    progress.end_stage()
    
    # ========== 阶段8: 主题偏向分析 ==========
    progress.start_stage(7)
    
    progress.log_subtask("初始化偏向分析器...")
    bias_analyzer = TopicBiasAnalyzer(config)
    
    progress.log_subtask("分析主题偏向...")
    bias_results = bias_analyzer.analyze_topic_bias(
        topic_results, structural_results, metadata_loader, filtered_docs
    )
    
    progress.log_subtask("生成偏向可视化...")
    bias_analyzer.create_bias_visualizations(bias_results, config.output_dir / 'topic_bias')
    
    # 偏向分析统计
    if 'organization_bias' in bias_results:
        progress.log_metric("组织偏向分析", f"{len(bias_results['organization_bias'])} 个组织")
    if 'temporal_bias' in bias_results:
        inc = len(bias_results['temporal_bias'].get('increasing_topics', []))
        dec = len(bias_results['temporal_bias'].get('decreasing_topics', []))
        progress.log_metric("时间趋势", f"上升{inc}个, 下降{dec}个主题")
    progress.end_stage()
    
    # ========== 阶段9: 时间序列分析 ==========
    progress.start_stage(8)
    time_series_analyzer = TimeSeriesAnalyzer(config)
    
    progress.log_subtask("分析时间序列...")
    time_series_results = time_series_analyzer.analyze_time_series(
        filtered_docs, metadata_loader, topic_results
    )
    
    progress.log_subtask("生成时间序列可视化...")
    time_series_analyzer.create_time_series_visualizations(
        time_series_results, config.output_dir / 'time_series'
    )
    
    # 时间序列统计
    ts_stats = time_series_results.get('statistics', {})
    progress.log_metric("年份范围", ts_stats.get('year_range', 'N/A'))
    progress.log_metric("总文档数", ts_stats.get('total_documents', 0))
    progress.log_metric("峰值年份", ts_stats.get('peak_year', 'N/A'))
    progress.log_metric("新兴标签", ts_stats.get('increasing_tags', 0))
    progress.log_metric("消退标签", ts_stats.get('decreasing_tags', 0))
    progress.end_stage()
    
    # ========== 阶段10: 保存结果 ==========
    progress.start_stage(9)
    
    final_results = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'config': {
            'n_documents': len(filtered_docs),
            'n_topics': config.n_topics,
            'model_type': config.topic_model_type,
            'year_range': f"{config.min_valid_year}-{config.max_valid_year}",
            'output_directory': str(config.output_dir),
            'filters': config.get_filter_summary()
        },
        'filter_statistics': {
            'countries': filter_stats['countries'],
            'org_types': filter_stats['org_types'],
            'years': filter_stats['years'],
            'sources': filter_stats['sources'],
            'doc_types': filter_stats['doc_types']
        },
        'narrative_analysis': {
            'n_documents_analyzed': len(narrative_results['document_arcs']),
            'global_patterns': narrative_results['global_patterns'],
            'arc_type_distribution': dict(Counter([arc['arc_type'] for arc in narrative_results['document_arcs'].values()]))
        },
        'discourse_analysis': {
            'n_documents': len(discourse_results.get('document_discourses', {})),
            'pattern_summary': discourse_results.get('pattern_summary', {})
        },
        'topic_analysis': {
            'topics': topic_results['topics'],
            'n_topics': len(topic_results['topics']),
            'document_topic_distribution': {
                'avg_topics_per_doc': np.mean([len(doc.get('topics', [])) for doc in filtered_docs]) if filtered_docs else 0
            }
        },
        'structural_effects': structural_results,
        'network_stats': network_stats,
        'bias_analysis': {
            'organization_bias': bias_results.get('organization_bias', {}),
            'temporal_trends': {
                k: v for k, v in bias_results.get('temporal_bias', {}).items() 
                if k != 'trends'
            }
        },
        'time_series_analysis': {
            'statistics': time_series_results.get('statistics', {}),
            'document_timeline': {
                'years': time_series_results.get('document_timeline', {}).get('years', []),
                'counts': time_series_results.get('document_timeline', {}).get('counts', []),
                'trend': time_series_results.get('document_timeline', {}).get('trend', 0)
            },
            'emerging_tags': time_series_results.get('tag_timeline', {}).get('increasing_tags', [])[:10],
            'fading_tags': time_series_results.get('tag_timeline', {}).get('decreasing_tags', [])[:10],
            'top_organizations': list(time_series_results.get('organization_timeline', {}).get('organization_trends', {}).keys())[:10]
        },
        'year_distribution': filter_stats['years'],
        'source_distribution': filter_stats['sources']
    }
    
    progress.log_subtask("保存JSON结果...")
    results_path = config.output_dir / 'analysis_results.json'
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, ensure_ascii=False, indent=2, cls=NumpyJSONEncoder)
    
    progress.log_metric("结果文件", results_path.name)
    progress.end_stage()
    
    # ========== 阶段11: 生成报告 ==========
    progress.start_stage(10)
    generate_analysis_report(final_results, config)
    
    # 显示输出文件列表
    output_files = list(config.output_dir.rglob("*"))
    n_files = len([f for f in output_files if f.is_file()])
    progress.log_metric("输出文件数", f"{n_files} 个")
    progress.end_stage()
    
    # 完成分析
    progress.finish()
    
    logger.info(f"\n📁 输出目录: {config.output_dir}")
    logger.info(f"📁 Gephi文件: {config.gephi_output_dir}")


def generate_analysis_report(results: Dict, config: NarrativeAnalysisConfig):
    """生成完备的分析报告"""
    report_path = config.output_dir / 'reports' / 'analysis_report.txt'
    
    with open(report_path, 'w', encoding='utf-8') as f:
        # 标题
        f.write("═" * 80 + "\n")
        f.write("叙事分析与语义网络分析报告\n")
        f.write("Narrative Analysis & Semantic Network Analysis Report\n")
        f.write("═" * 80 + "\n\n")
        
        f.write(f"生成时间: {results['timestamp']}\n")
        f.write(f"输出目录: {results['config'].get('output_directory', 'N/A')}\n\n")
        
        # ========== 筛选条件 ==========
        filters = results['config'].get('filters', {})
        if any(k for k in filters.keys() if k != 'year_range'):
            f.write("═" * 80 + "\n")
            f.write("🔍 数据筛选条件\n")
            f.write("═" * 80 + "\n\n")
            
            if filters.get('countries'):
                f.write(f"  🌍 国家筛选: {', '.join(filters['countries'])}\n")
            if filters.get('org_types'):
                f.write(f"  🏢 组织类型筛选: {', '.join(filters['org_types'])}\n")
            if filters.get('doc_types'):
                f.write(f"  📄 文档类型筛选: {', '.join(filters['doc_types'])}\n")
            if filters.get('data_sources'):
                f.write(f"  📁 数据源筛选: {', '.join(filters['data_sources'])}\n")
            f.write("\n")
        
        # ========== 数据概览 ==========
        f.write("═" * 80 + "\n")
        f.write("📊 数据概览\n")
        f.write("═" * 80 + "\n\n")
        
        f.write(f"  ├─ 文档总数: {results['config']['n_documents']}\n")
        f.write(f"  ├─ 主题数量: {results['config']['n_topics']}\n")
        f.write(f"  ├─ 主题模型: {results['config']['model_type'].upper()}\n")
        f.write(f"  └─ 年份范围: {results['config'].get('year_range', 'N/A')}\n\n")
        
        # 国家分布
        if 'filter_statistics' in results and results['filter_statistics'].get('countries'):
            countries = results['filter_statistics']['countries']
            if countries:
                f.write("  🌍 国家/地区分布:\n")
                for country, count in sorted(countries.items(), key=lambda x: x[1], reverse=True)[:10]:
                    f.write(f"    - {country}: {count} 个文档\n")
                f.write("\n")
        
        # 组织类型分布
        if 'filter_statistics' in results and results['filter_statistics'].get('org_types'):
            org_types = results['filter_statistics']['org_types']
            if org_types:
                f.write("  🏢 组织类型分布:\n")
                for org_type, count in sorted(org_types.items(), key=lambda x: x[1], reverse=True)[:10]:
                    f.write(f"    - {org_type}: {count} 个文档\n")
                f.write("\n")
        
        # 年份分布
        if 'year_distribution' in results:
            f.write("  📅 年份分布:\n")
            year_dist = results['year_distribution']
            for year in sorted(year_dist.keys()):
                count = year_dist[year]
                bar = '█' * min(int(count / max(year_dist.values()) * 20), 20)
                f.write(f"    {year}: {bar} ({count})\n")
            f.write("\n")
        
        # 数据源分布
        if 'source_distribution' in results:
            f.write("  📁 数据源分布:\n")
            for source, count in results['source_distribution'].items():
                f.write(f"    - {source}: {count} 个文档\n")
            f.write("\n")
        
        # ========== 叙事分析结果 ==========
        f.write("═" * 80 + "\n")
        f.write("📖 叙事分析结果\n")
        f.write("═" * 80 + "\n\n")
        
        narrative = results['narrative_analysis']
        f.write(f"  ├─ 分析文档数: {narrative['n_documents_analyzed']}\n")
        
        if 'global_patterns' in narrative:
            patterns = narrative['global_patterns']
            if 'avg_semantic_shift' in patterns:
                f.write(f"  ├─ 平均语义漂移: {patterns['avg_semantic_shift']:.4f}\n")
            if 'arc_distribution' in patterns:
                f.write("  └─ 叙事弧类型分布:\n")
                for arc_type, count in patterns['arc_distribution'].items():
                    pct = count / narrative['n_documents_analyzed'] * 100 if narrative['n_documents_analyzed'] > 0 else 0
                    f.write(f"      - {arc_type}: {count} ({pct:.1f}%)\n")
        
        # 叙事弧类型详细分布
        if 'arc_type_distribution' in narrative:
            f.write("\n  📈 叙事弧类型详情:\n")
            for arc_type, count in narrative['arc_type_distribution'].items():
                f.write(f"    - {arc_type}: {count} 个文档\n")
        f.write("\n")
        
        # ========== 话语分析结果 ==========
        if 'discourse_analysis' in results:
            f.write("═" * 80 + "\n")
            f.write("💬 话语分析结果\n")
            f.write("═" * 80 + "\n\n")
            
            discourse = results['discourse_analysis']
            f.write(f"  ├─ 分析文档数: {discourse.get('n_documents', 0)}\n")
            if 'pattern_summary' in discourse:
                f.write("  └─ 话语模式摘要: 已完成分析\n")
            f.write("\n")
        
        # ========== 主题分析结果 ==========
        f.write("═" * 80 + "\n")
        f.write("🎯 主题分析结果\n")
        f.write("═" * 80 + "\n\n")
        
        topics = results['topic_analysis']
        f.write(f"  主题总数: {topics['n_topics']}\n\n")
        
        for topic_id, topic_data in sorted(topics['topics'].items()):
            f.write(f"  【Topic {topic_id}】 {topic_data['name']}\n")
            keywords = topic_data.get('keywords', [])
            if keywords:
                f.write(f"    关键词: {', '.join(keywords[:15])}\n")
            if 'weight' in topic_data:
                f.write(f"    权重: {topic_data['weight']:.4f}\n")
            f.write("\n")
        
        # 文档主题分布统计
        if 'document_topic_distribution' in topics:
            dist = topics['document_topic_distribution']
            f.write(f"  📊 文档主题分布统计:\n")
            f.write(f"    平均每文档主题数: {dist.get('avg_topics_per_doc', 0):.2f}\n\n")
        
        # ========== 网络分析结果 ==========
        f.write("═" * 80 + "\n")
        f.write("🕸️ 网络分析结果\n")
        f.write("═" * 80 + "\n\n")
        
        stats = results['network_stats']
        if stats and 'basic' in stats:
            basic = stats['basic']
            f.write("  基本统计:\n")
            f.write(f"    ├─ 节点数: {basic.get('n_nodes', 0)}\n")
            f.write(f"    ├─ 边数: {basic.get('n_edges', 0)}\n")
            f.write(f"    └─ 网络密度: {basic.get('density', 0):.4f}\n\n")
        
        if stats and 'clustering' in stats:
            clustering = stats['clustering']
            f.write("  聚类统计:\n")
            f.write(f"    └─ 平均聚类系数: {clustering.get('avg_clustering', 0):.4f}\n\n")
        
        if stats and 'centrality' in stats:
            centrality = stats['centrality']
            f.write("  中心性统计:\n")
            f.write(f"    ├─ 最大度中心性: {centrality.get('max_degree', 0)}\n")
            f.write(f"    └─ 最大介数中心性: {centrality.get('max_betweenness', 0):.4f}\n\n")
        
        # ========== 主题偏向分析 ==========
        f.write("═" * 80 + "\n")
        f.write("⚖️ 主题偏向分析\n")
        f.write("═" * 80 + "\n\n")
        
        bias = results['bias_analysis']
        
        # 时间趋势
        if 'temporal_trends' in bias:
            trends = bias['temporal_trends']
            f.write("  时间趋势:\n")
            inc = trends.get('increasing_topics', [])
            dec = trends.get('decreasing_topics', [])
            stable = trends.get('stable_topics', [])
            f.write(f"    ├─ 上升趋势: {len(inc)} 个主题\n")
            f.write(f"    ├─ 下降趋势: {len(dec)} 个主题\n")
            f.write(f"    └─ 稳定趋势: {len(stable)} 个主题\n\n")
        
        # 组织偏向
        if 'organization_bias' in bias and bias['organization_bias']:
            org_bias = bias['organization_bias']
            f.write(f"  组织偏向分析: {len(org_bias)} 个组织\n")
            # 显示前10个组织
            for i, (org, data) in enumerate(list(org_bias.items())[:10]):
                f.write(f"    {i+1}. {org}\n")
            f.write("\n")
        
        # ========== 时间序列分析 ==========
        f.write("═" * 80 + "\n")
        f.write("📈 时间序列分析\n")
        f.write("═" * 80 + "\n\n")
        
        if 'time_series_analysis' in results:
            ts = results['time_series_analysis']
            stats = ts.get('statistics', {})
            
            f.write("  统计摘要:\n")
            f.write(f"    ├─ 年份范围: {stats.get('year_range', 'N/A')}\n")
            f.write(f"    ├─ 总文档数: {stats.get('total_documents', 0)}\n")
            f.write(f"    ├─ 年均文档数: {stats.get('avg_documents_per_year', 0):.1f}\n")
            f.write(f"    ├─ 峰值年份: {stats.get('peak_year', 'N/A')} ({stats.get('peak_documents', 0)} 文档)\n")
            f.write(f"    └─ 唯一标签数: {stats.get('unique_tags', 0)}\n\n")
            
            # 文档时间线
            doc_timeline = ts.get('document_timeline', {})
            if doc_timeline.get('years'):
                f.write("  文档时间线:\n")
                years = doc_timeline.get('years', [])
                counts = doc_timeline.get('counts', [])
                for year, count in zip(years[-10:], counts[-10:]):  # 只显示最近10年
                    bar = '█' * min(int(count / max(counts) * 20) if counts else 0, 20)
                    f.write(f"    {year}: {bar} ({count})\n")
                f.write("\n")
            
            # 新兴标签
            emerging = ts.get('emerging_tags', [])
            if emerging:
                f.write("  📈 新兴标签 (Top 10):\n")
                for tag, trend in emerging[:10]:
                    f.write(f"    • {tag}: {trend:.3f}\n")
                f.write("\n")
            
            # 消退标签
            fading = ts.get('fading_tags', [])
            if fading:
                f.write("  📉 消退标签 (Top 10):\n")
                for tag, trend in fading[:10]:
                    f.write(f"    • {tag}: {trend:.3f}\n")
                f.write("\n")
            
            # 活跃组织
            top_orgs = ts.get('top_organizations', [])
            if top_orgs:
                f.write(f"  🏢 活跃组织 (Top 10):\n")
                for i, org in enumerate(top_orgs[:10], 1):
                    f.write(f"    {i}. {org}\n")
                f.write("\n")
        
        # ========== 输出文件列表 ==========
        f.write("═" * 80 + "\n")
        f.write("📁 输出文件\n")
        f.write("═" * 80 + "\n\n")
        
        f.write(f"  ├─ JSON结果: analysis_results.json\n")
        f.write(f"  ├─ Gephi文件: {config.gephi_output_dir}\n")
        f.write(f"  ├─ 可视化文件: {config.output_dir / 'visualizations'}\n")
        f.write(f"  ├─ 叙事分析: {config.output_dir / 'narrative'}\n")
        f.write(f"  ├─ 语义网络: {config.output_dir / 'semantic_network'}\n")
        f.write(f"  ├─ 主题偏向: {config.output_dir / 'topic_bias'}\n")
        f.write(f"  └─ 时间序列: {config.output_dir / 'time_series'}\n\n")
        
        # 列出所有生成的文件
        f.write("  📋 生成的文件列表:\n")
        for subdir in ['visualizations', 'narrative', 'semantic_network', 'topic_bias', 'time_series', 'reports']:
            subdir_path = config.output_dir / subdir
            if subdir_path.exists():
                files = list(subdir_path.glob("*"))
                for file in files:
                    if file.is_file() and not file.name.startswith('.'):
                        size = file.stat().st_size
                        size_str = f"{size/1024:.1f}KB" if size > 1024 else f"{size}B"
                        f.write(f"    - {subdir}/{file.name} ({size_str})\n")
        
        f.write("\n" + "═" * 80 + "\n")
        f.write("报告生成完成\n")
        f.write("═" * 80 + "\n")
    
    logger.info(f"✅ 分析报告已保存: {report_path}")


class NumpyJSONEncoder(json.JSONEncoder):
    """自定义JSON编码器，处理numpy类型"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# ==================== 筛选分析示例 ====================
def run_analysis_with_country_filter(countries: List[str]):
    """运行带有国家筛选的分析
    
    Args:
        countries: 要分析的国家列表，如 ['美国'] 或 ['美国', '中国', '英国']
    
    示例:
        # 只分析美国的文档
        run_analysis_with_country_filter(['美国'])
        
        # 分析美国和中国的文档
        run_analysis_with_country_filter(['美国', '中国'])
    """
    global progress
    
    logger.info("═" * 80)
    logger.info("🚀 叙事与语义网络分析系统启动 (带国家筛选)")
    logger.info(f"🌍 筛选国家: {countries}")
    logger.info(f"📅 开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("═" * 80)
    
    # 初始化配置并设置筛选条件
    progress.start_stage(0)
    config = NarrativeAnalysisConfig()
    config.set_country_filter(countries)
    progress.log_metric("输出目录", config.output_dir)
    progress.log_metric("筛选国家", countries)
    progress.end_stage()
    
    # 继续执行后续分析流程...
    # 加载停用词
    progress.start_stage(1)
    stopwords_loader = StopwordsLoader(config.stopwords_paths)
    stop_words = stopwords_loader.load_all_stopwords()
    progress.end_stage()
    
    # 加载元数据
    progress.start_stage(2)
    metadata_loader = MetadataLoader(config.metadata_dir)
    progress.log_metric("可用国家", metadata_loader.get_all_countries()[:10])
    progress.end_stage()
    
    # 加载文档并应用筛选
    progress.start_stage(3)
    doc_loader = DocumentLoader(
        config.agora_fulltext_dir,
        config.original_data_dir,
        metadata_loader
    )
    documents = doc_loader.load_documents()
    
    if not documents:
        logger.error("❌ 没有找到有效文档")
        return
    
    filtered_docs, filter_stats = apply_filters(documents, metadata_loader, config)
    
    progress.log_metric("原始文档数", f"{len(documents)} 个")
    progress.log_metric(f"{countries} 文档数", f"{len(filtered_docs)} 个")
    
    if len(filtered_docs) == 0:
        logger.error(f"❌ 没有找到 {countries} 的文档")
        logger.info(f"💡 可用国家: {metadata_loader.get_all_countries()}")
        return
    
    progress.end_stage()
    
    # 执行完整分析
    _execute_analysis(filtered_docs, metadata_loader, config, stop_words, filter_stats)


def run_analysis_with_org_type_filter(org_types: List[str]):
    """运行带有组织类型筛选的分析
    
    Args:
        org_types: 要分析的组织类型列表，如 ['government'] 或 ['government', 'tech_company']
    
    示例:
        # 只分析政府文档
        run_analysis_with_org_type_filter(['government'])
        
        # 分析政府和科技公司文档
        run_analysis_with_org_type_filter(['government', 'tech_company'])
    """
    global progress
    
    logger.info("═" * 80)
    logger.info("🚀 叙事与语义网络分析系统启动 (带组织类型筛选)")
    logger.info(f"🏢 筛选组织类型: {org_types}")
    logger.info(f"📅 开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("═" * 80)
    
    # 初始化配置并设置筛选条件
    progress.start_stage(0)
    config = NarrativeAnalysisConfig()
    config.set_org_type_filter(org_types)
    progress.log_metric("输出目录", config.output_dir)
    progress.log_metric("筛选组织类型", org_types)
    progress.end_stage()
    
    # 加载停用词
    progress.start_stage(1)
    stopwords_loader = StopwordsLoader(config.stopwords_paths)
    stop_words = stopwords_loader.load_all_stopwords()
    progress.end_stage()
    
    # 加载元数据
    progress.start_stage(2)
    metadata_loader = MetadataLoader(config.metadata_dir)
    progress.log_metric("可用组织类型", metadata_loader.get_all_org_types())
    progress.end_stage()
    
    # 加载文档并应用筛选
    progress.start_stage(3)
    doc_loader = DocumentLoader(
        config.agora_fulltext_dir,
        config.original_data_dir,
        metadata_loader
    )
    documents = doc_loader.load_documents()
    
    if not documents:
        logger.error("❌ 没有找到有效文档")
        return
    
    filtered_docs, filter_stats = apply_filters(documents, metadata_loader, config)
    
    progress.log_metric("原始文档数", f"{len(documents)} 个")
    progress.log_metric(f"{org_types} 文档数", f"{len(filtered_docs)} 个")
    
    if len(filtered_docs) == 0:
        logger.error(f"❌ 没有找到 {org_types} 的文档")
        logger.info(f"💡 可用组织类型: {metadata_loader.get_all_org_types()}")
        return
    
    progress.end_stage()
    
    # 执行完整分析
    _execute_analysis(filtered_docs, metadata_loader, config, stop_words, filter_stats)


def _execute_analysis(filtered_docs, metadata_loader, config, stop_words, filter_stats):
    """内部函数：执行完整分析流程"""
    # 这里可以继续执行完整的分析流程
    # 为简化示例，这里只执行部分分析
    
    logger.info(f"✅ 筛选完成，共 {len(filtered_docs)} 个文档")
    logger.info(f"📁 输出目录: {config.output_dir}")
    
    # 后续可以调用 run_full_analysis() 中的后续步骤
    # 或者直接调用该函数并传入筛选后的文档


def list_available_filters():
    """列出所有可用的筛选选项"""
    logger.info("═" * 80)
    logger.info("📋 可用筛选选项查询")
    logger.info("═" * 80)
    
    # 加载元数据
    metadata_loader = MetadataLoader(Path("/Volumes/ZimingYe/A_project/12月数据采集汇总/数据标注/1226标注结果"))
    stats = metadata_loader.get_statistics()
    
    logger.info("\n🌍 可用国家/地区:")
    for country, count in stats['countries'].items():
        logger.info(f"  - {country}: {count} 个文档")
    
    logger.info("\n🏢 可用组织类型:")
    for org_type, count in stats['org_types'].items():
        logger.info(f"  - {org_type}: {count} 个文档")
    
    logger.info("\n📅 年份分布:")
    for year, count in stats['years'].items():
        logger.info(f"  - {year}: {count} 个文档")
    
    logger.info("\n═" * 80)


# ==================== 中美欧对比分析系统 ====================
class ComparativeNetworkAnalyzer:
    """多地区语义网络对比分析器
    
    使用权威的语义网络对比方法进行交叉对比分析：
    1. 网络拓扑指标对比（节点、边、密度、聚类系数、平均路径长度）
    2. 节点重叠分析（Jaccard相似度、共有/特有节点）
    3. 边重叠分析（共有/特有边、权重对比）
    4. 核心概念对比（度中心性、介数中心性排名）
    5. 社区结构对比（社区划分、主题差异）
    6. 网络对齐分析（Graph Matching）
    """
    
    def __init__(self, config: NarrativeAnalysisConfig):
        self.config = config
        self.stop_words = set()
        self.region_networks = {}  # 存储各地区的网络
        self.region_stats = {}     # 存储各地区的统计指标
        
    def set_stop_words(self, stop_words: Set[str]):
        """设置停用词"""
        self.stop_words = stop_words
    
    def build_regional_networks(self, region_docs: Dict[str, List[Dict]]) -> Dict[str, nx.Graph]:
        """为每个地区构建语义网络
        
        Args:
            region_docs: 地区文档字典 {'美国': [docs], '中国': [docs], '欧盟': [docs]}
        
        Returns:
            地区网络字典 {'美国': G1, '中国': G2, '欧盟': G3}
        """
        logger.info("\n📊 阶段1: 构建各地区语义网络")
        logger.info("─" * 60)
        
        for region, docs in region_docs.items():
            if len(docs) < 5:
                logger.warning(f"  ⚠️ {region} 文档数不足({len(docs)}个)，跳过")
                continue
            
            logger.info(f"\n  🌍 构建 {region} 语义网络 ({len(docs)} 个文档)...")
            
            # 使用TF-IDF提取关键词
            vectorizer = TfidfVectorizer(
                max_features=300,
                stop_words=list(self.stop_words),
                ngram_range=(1, 2),
                min_df=2,
                token_pattern=r'(?u)\b(?![0-9]+\b)[a-zA-Z][a-zA-Z0-9]{2,}\b'
            )
            
            all_texts = [doc['content'] for doc in docs]
            try:
                tfidf_matrix = vectorizer.fit_transform(all_texts)
                feature_names = list(vectorizer.get_feature_names_out())
            except:
                logger.warning(f"  ⚠️ {region} TF-IDF提取失败，跳过")
                continue
            
            # 过滤数字词
            feature_names = [f for f in feature_names if not self._is_numeric_word(f)]
            
            # 计算共现
            co_occurrence = defaultdict(lambda: defaultdict(int))
            for doc in docs:
                content = doc['content'].lower()
                words_in_doc = set()
                for term in feature_names:
                    if term in content:
                        words_in_doc.add(term)
                for w1 in words_in_doc:
                    for w2 in words_in_doc:
                        if w1 < w2:
                            co_occurrence[w1][w2] += 1
            
            # 构建网络
            G = nx.Graph()
            for w1, neighbors in co_occurrence.items():
                for w2, count in neighbors.items():
                    if count >= 2:
                        weight = count / len(docs)
                        if weight >= 0.05:
                            G.add_edge(w1, w2, weight=float(weight), co_occurrence=int(count))
            
            # 添加节点属性
            term_scores = np.array(tfidf_matrix.sum(axis=0)).flatten()
            term_score_dict = {feature_names[i]: float(term_scores[i]) for i in range(len(feature_names))}
            for node in G.nodes():
                G.nodes[node]['tfidf_score'] = term_score_dict.get(node, 0)
                G.nodes[node]['degree'] = G.degree(node)
                G.nodes[node]['region'] = region
            
            self.region_networks[region] = G
            self.region_stats[region] = self._calculate_network_stats(G)
            
            logger.info(f"  ✓ {region}: {G.number_of_nodes()} 节点, {G.number_of_edges()} 边")
        
        return self.region_networks
    
    def _is_numeric_word(self, word: str) -> bool:
        """检查是否为数字词"""
        word_lower = word.lower()
        if word_lower.isdigit():
            return True
        if re.match(r'^[0-9]+[a-zA-Z]*$', word_lower):
            return True
        if re.search(r'[0-9]{3,}', word_lower):
            return True
        return False
    
    def _calculate_network_stats(self, G: nx.Graph) -> Dict:
        """计算网络统计指标"""
        stats = {
            'n_nodes': G.number_of_nodes(),
            'n_edges': G.number_of_edges(),
            'density': nx.density(G) if G.number_of_nodes() > 0 else 0,
        }
        
        if G.number_of_nodes() > 1:
            stats['avg_clustering'] = nx.average_clustering(G)
            stats['transitivity'] = nx.transitivity(G)
            
            if nx.is_connected(G):
                stats['avg_path_length'] = nx.average_shortest_path_length(G)
                stats['diameter'] = nx.diameter(G)
            else:
                # 取最大连通分量
                largest_cc = max(nx.connected_components(G), key=len)
                G_lcc = G.subgraph(largest_cc)
                stats['avg_path_length'] = nx.average_shortest_path_length(G_lcc) if G_lcc.number_of_nodes() > 1 else 0
                stats['diameter'] = nx.diameter(G_lcc) if G_lcc.number_of_nodes() > 1 else 0
                stats['n_components'] = nx.number_connected_components(G)
            
            # 度中心性
            degree_cent = nx.degree_centrality(G)
            stats['top_degree_nodes'] = sorted(degree_cent.items(), key=lambda x: x[1], reverse=True)[:20]
            
            # 介数中心性（限制计算量）
            if G.number_of_nodes() < 200:
                betweenness = nx.betweenness_centrality(G)
                stats['top_betweenness_nodes'] = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:20]
            
            # 社区检测
            try:
                communities = list(nx.community.greedy_modularity_communities(G))
                stats['n_communities'] = len(communities)
                stats['community_sizes'] = [len(c) for c in communities[:10]]
                stats['modularity'] = nx.community.modularity(G, communities)
            except:
                stats['n_communities'] = 0
                stats['modularity'] = 0
        
        return stats
    
    def compare_networks(self) -> Dict:
        """网络对比分析
        
        Returns:
            对比结果字典
        """
        logger.info("\n📊 阶段2: 网络对比分析")
        logger.info("─" * 60)
        
        regions = list(self.region_networks.keys())
        if len(regions) < 2:
            logger.warning("  ⚠️ 至少需要2个地区的网络才能进行对比")
            return {}
        
        comparison_results = {
            'topology_comparison': self._compare_topology(),
            'node_overlap': self._analyze_node_overlap(regions),
            'edge_overlap': self._analyze_edge_overlap(regions),
            'core_concepts_comparison': self._compare_core_concepts(regions),
            'community_comparison': self._compare_communities(regions),
            'network_similarity': self._calculate_network_similarity(regions)
        }
        
        return comparison_results
    
    def _compare_topology(self) -> Dict:
        """对比网络拓扑指标"""
        logger.info("\n  📐 网络拓扑指标对比...")
        
        topology_data = []
        for region, stats in self.region_stats.items():
            topology_data.append({
                'region': region,
                '节点数': stats.get('n_nodes', 0),
                '边数': stats.get('n_edges', 0),
                '密度': round(stats.get('density', 0), 4),
                '平均聚类系数': round(stats.get('avg_clustering', 0), 4),
                '传递性': round(stats.get('transitivity', 0), 4),
                '平均路径长度': round(stats.get('avg_path_length', 0), 4),
                '直径': stats.get('diameter', 0),
                '社区数': stats.get('n_communities', 0),
                '模块度': round(stats.get('modularity', 0), 4)
            })
        
        df = pd.DataFrame(topology_data)
        logger.info(f"\n{df.to_string(index=False)}")
        
        return {
            'dataframe': df.to_dict('records'),
            'summary': {
                region: {k: v for k, v in stats.items() if k != 'top_degree_nodes' and k != 'top_betweenness_nodes'}
                for region, stats in self.region_stats.items()
            }
        }
    
    def _analyze_node_overlap(self, regions: List[str]) -> Dict:
        """分析节点重叠"""
        logger.info("\n  🔗 节点重叠分析...")
        
        # 获取各地区的节点集合
        node_sets = {r: set(self.region_networks[r].nodes()) for r in regions}
        
        # 计算两两Jaccard相似度
        jaccard_matrix = {}
        pairwise_overlap = {}
        
        for i, r1 in enumerate(regions):
            jaccard_matrix[r1] = {}
            pairwise_overlap[r1] = {}
            for j, r2 in enumerate(regions):
                if i < j:
                    intersection = node_sets[r1] & node_sets[r2]
                    union = node_sets[r1] | node_sets[r2]
                    jaccard = len(intersection) / len(union) if union else 0
                    
                    jaccard_matrix[r1][r2] = round(jaccard, 4)
                    pairwise_overlap[r1][r2] = {
                        'intersection_size': len(intersection),
                        'union_size': len(union),
                        'jaccard_similarity': round(jaccard, 4),
                        'r1_unique': len(node_sets[r1] - node_sets[r2]),
                        'r2_unique': len(node_sets[r2] - node_sets[r1]),
                        'common_nodes': list(intersection)[:50]  # 保存前50个共同节点
                    }
                    
                    logger.info(f"    {r1} vs {r2}: Jaccard={jaccard:.4f}, 共同节点={len(intersection)}")
        
        # 三地区共同节点
        if len(regions) >= 3:
            common_all = node_sets[regions[0]]
            for r in regions[1:]:
                common_all = common_all & node_sets[r]
            logger.info(f"\n    三地区共同节点数: {len(common_all)}")
        else:
            common_all = set()
        
        return {
            'jaccard_matrix': jaccard_matrix,
            'pairwise_overlap': pairwise_overlap,
            'common_to_all': list(common_all),
            'unique_nodes': {
                r: list(node_sets[r] - set().union(*[node_sets[other] for other in regions if other != r]))[:30]
                for r in regions
            }
        }
    
    def _analyze_edge_overlap(self, regions: List[str]) -> Dict:
        """分析边重叠"""
        logger.info("\n  🔗 边重叠分析...")
        
        # 获取各地区的边集合（标准化为有序元组）
        edge_sets = {}
        for r in regions:
            edges = set()
            for u, v in self.region_networks[r].edges():
                edges.add(tuple(sorted([u, v])))
            edge_sets[r] = edges
        
        # 计算两两边重叠
        edge_overlap = {}
        for i, r1 in enumerate(regions):
            edge_overlap[r1] = {}
            for j, r2 in enumerate(regions):
                if i < j:
                    intersection = edge_sets[r1] & edge_sets[r2]
                    union = edge_sets[r1] | edge_sets[r2]
                    jaccard = len(intersection) / len(union) if union else 0
                    
                    edge_overlap[r1][r2] = {
                        'intersection_size': len(intersection),
                        'jaccard_similarity': round(jaccard, 4),
                        'common_edges': [f"{e[0]}-{e[1]}" for e in list(intersection)[:30]]
                    }
                    
                    logger.info(f"    {r1} vs {r2}: 边Jaccard={jaccard:.4f}, 共同边={len(intersection)}")
        
        return edge_overlap
    
    def _compare_core_concepts(self, regions: List[str]) -> Dict:
        """对比核心概念"""
        logger.info("\n  ⭐ 核心概念对比...")
        
        core_concepts = {}
        
        for region in regions:
            stats = self.region_stats.get(region, {})
            top_nodes = stats.get('top_degree_nodes', [])[:15]
            
            core_concepts[region] = {
                'top_by_degree': [(node, round(score, 4)) for node, score in top_nodes],
                'unique_core': []
            }
            
            logger.info(f"\n  {region} 核心概念 (度中心性Top10):")
            for node, score in top_nodes[:10]:
                logger.info(f"    - {node}: {score:.4f}")
        
        # 找出各地区特有的核心概念
        all_core = {r: set([n[0] for n in core_concepts[r]['top_by_degree']]) for r in regions}
        for r in regions:
            other_cores = set().union(*[all_core[other] for other in regions if other != r])
            unique = all_core[r] - other_cores
            core_concepts[r]['unique_core'] = list(unique)
            if unique:
                logger.info(f"\n  {r} 特有核心概念: {', '.join(list(unique)[:10])}")
        
        return core_concepts
    
    def _compare_communities(self, regions: List[str]) -> Dict:
        """对比社区结构"""
        logger.info("\n  🏘️ 社区结构对比...")
        
        community_comparison = {}
        
        for region in regions:
            G = self.region_networks[region]
            try:
                communities = list(nx.community.greedy_modularity_communities(G))
                
                # 获取每个社区的关键词
                community_keywords = []
                for i, comm in enumerate(communities[:5]):  # 取前5个社区
                    subG = G.subgraph(comm)
                    # 社区内度最高的节点作为代表
                    degrees = dict(subG.degree())
                    top_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:5]
                    community_keywords.append({
                        'community_id': i,
                        'size': len(comm),
                        'top_keywords': [n[0] for n in top_nodes]
                    })
                
                community_comparison[region] = {
                    'n_communities': len(communities),
                    'modularity': nx.community.modularity(G, communities),
                    'top_communities': community_keywords
                }
                
                logger.info(f"  {region}: {len(communities)} 个社区, 模块度={nx.community.modularity(G, communities):.4f}")
            except Exception as e:
                community_comparison[region] = {'error': str(e)}
        
        return community_comparison
    
    def _calculate_network_similarity(self, regions: List[str]) -> Dict:
        """计算网络整体相似度"""
        logger.info("\n  📊 网络整体相似度计算...")
        
        # 使用多种方法计算网络相似度
        similarity_results = {}
        
        for i, r1 in enumerate(regions):
            similarity_results[r1] = {}
            for j, r2 in enumerate(regions):
                if i < j:
                    G1, G2 = self.region_networks[r1], self.region_networks[r2]
                    
                    # 1. 节点Jaccard相似度
                    nodes1, nodes2 = set(G1.nodes()), set(G2.nodes())
                    node_jaccard = len(nodes1 & nodes2) / len(nodes1 | nodes2) if (nodes1 | nodes2) else 0
                    
                    # 2. 结构相似度（基于度分布）
                    deg1 = sorted([d for n, d in G1.degree()])
                    deg2 = sorted([d for n, d in G2.degree()])
                    # 归一化后计算相关性
                    max_len = max(len(deg1), len(deg2))
                    deg1_norm = deg1 + [0] * (max_len - len(deg1))
                    deg2_norm = deg2 + [0] * (max_len - len(deg2))
                    if max_len > 0:
                        degree_corr = np.corrcoef(deg1_norm, deg2_norm)[0, 1]
                    else:
                        degree_corr = 0
                    
                    # 3. 密度相似度
                    density1 = nx.density(G1)
                    density2 = nx.density(G2)
                    density_sim = 1 - abs(density1 - density2)
                    
                    # 综合相似度
                    overall_sim = 0.4 * node_jaccard + 0.3 * abs(degree_corr) + 0.3 * density_sim
                    
                    similarity_results[r1][r2] = {
                        'node_jaccard': round(node_jaccard, 4),
                        'degree_correlation': round(degree_corr, 4),
                        'density_similarity': round(density_sim, 4),
                        'overall_similarity': round(overall_sim, 4)
                    }
                    
                    logger.info(f"  {r1} vs {r2}: 综合相似度={overall_sim:.4f}")
        
        return similarity_results
    
    def create_comparison_visualizations(self, comparison_results: Dict, output_dir: Path):
        """创建对比可视化"""
        logger.info("\n📊 阶段3: 生成对比可视化")
        logger.info("─" * 60)
        
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # 1. 拓扑指标对比雷达图
        self._create_topology_radar(comparison_results.get('topology_comparison', {}), output_dir)
        
        # 2. 节点重叠热力图
        self._create_overlap_heatmap(comparison_results.get('node_overlap', {}), output_dir)
        
        # 3. 核心概念对比图
        self._create_core_concepts_chart(comparison_results.get('core_concepts_comparison', {}), output_dir)
        
        # 4. 网络相似度矩阵
        self._create_similarity_matrix(comparison_results.get('network_similarity', {}), output_dir)
        
        # 5. 三地区韦恩图（共同/特有节点）
        self._create_venn_diagram(comparison_results.get('node_overlap', {}), output_dir)
        
        logger.info(f"  ✓ 可视化文件已保存至: {output_dir}")
    
    def _create_topology_radar(self, topology_data: Dict, output_dir: Path):
        """创建拓扑指标雷达图"""
        records = topology_data.get('dataframe', [])
        if not records:
            return
        
        # 标准化指标
        df = pd.DataFrame(records)
        metrics = ['节点数', '边数', '密度', '平均聚类系数', '模块度']
        
        fig = go.Figure()
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        
        for idx, row in df.iterrows():
            values = [row.get(m, 0) for m in metrics]
            # 归一化
            max_vals = [df[m].max() if df[m].max() > 0 else 1 for m in metrics]
            normalized = [v/m if m > 0 else 0 for v, m in zip(values, max_vals)]
            normalized.append(normalized[0])  # 闭合
            
            fig.add_trace(go.Scatterpolar(
                r=normalized,
                theta=metrics + [metrics[0]],
                fill='toself',
                name=row['region'],
                line_color=colors[idx % len(colors)]
            ))
        
        fig.update_layout(
            title='网络拓扑指标对比 (归一化)<br><sup>(Network Topology Comparison)</sup>',
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=True,
            paper_bgcolor='#F5FAFF',
            font=dict(color='#1565C0')
        )
        
        output_path = output_dir / 'topology_radar.html'
        fig.write_html(str(output_path))
        logger.info(f"  ✓ 拓扑雷达图: topology_radar.html")
    
    def _create_overlap_heatmap(self, node_overlap: Dict, output_dir: Path):
        """创建节点重叠热力图"""
        jaccard = node_overlap.get('jaccard_matrix', {})
        if not jaccard:
            return
        
        regions = list(jaccard.keys()) + [list(jaccard[list(jaccard.keys())[0]].keys())[0]] if jaccard else []
        
        # 构建矩阵
        n = len(regions)
        matrix = np.zeros((n, n))
        for i, r1 in enumerate(regions):
            for j, r2 in enumerate(regions):
                if r1 in jaccard and r2 in jaccard[r1]:
                    matrix[i, j] = jaccard[r1][r2]
                elif i == j:
                    matrix[i, j] = 1.0
                elif r2 in jaccard and r1 in jaccard.get(r2, {}):
                    matrix[i, j] = jaccard[r2][r1]
        
        fig = go.Figure(data=go.Heatmap(
            z=matrix,
            x=regions,
            y=regions,
            colorscale='Blues',
            hovertemplate='%{x} vs %{y}<br>Jaccard: %{z:.4f}<extra></extra>',
            text=[[f'{matrix[i][j]:.3f}' for j in range(n)] for i in range(n)],
            texttemplate='%{text}'
        ))
        
        fig.update_layout(
            title='节点Jaccard相似度矩阵<br><sup>(Node Jaccard Similarity Matrix)</sup>',
            paper_bgcolor='#F5FAFF',
            font=dict(color='#1565C0')
        )
        
        output_path = output_dir / 'node_overlap_heatmap.html'
        fig.write_html(str(output_path))
        logger.info(f"  ✓ 节点重叠热力图: node_overlap_heatmap.html")
    
    def _create_core_concepts_chart(self, core_concepts: Dict, output_dir: Path):
        """创建核心概念对比图"""
        if not core_concepts:
            return
        
        fig = make_subplots(
            rows=1, cols=len(core_concepts),
            subplot_titles=list(core_concepts.keys()),
            specs=[[{'type': 'bar'} for _ in core_concepts]]
        )
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        
        for idx, (region, data) in enumerate(core_concepts.items()):
            top_nodes = data.get('top_by_degree', [])[:10]
            if top_nodes:
                nodes = [n[0] for n in top_nodes]
                scores = [n[1] for n in top_nodes]
                
                fig.add_trace(
                    go.Bar(
                        y=nodes[::-1],
                        x=scores[::-1],
                        orientation='h',
                        marker_color=colors[idx % len(colors)],
                        name=region,
                        showlegend=False
                    ),
                    row=1, col=idx+1
                )
        
        fig.update_layout(
            title='各地区核心概念对比 (度中心性Top10)<br><sup>(Core Concepts Comparison by Degree Centrality)</sup>',
            height=500,
            paper_bgcolor='#F5FAFF',
            font=dict(color='#1565C0')
        )
        
        output_path = output_dir / 'core_concepts_comparison.html'
        fig.write_html(str(output_path))
        logger.info(f"  ✓ 核心概念对比图: core_concepts_comparison.html")
    
    def _create_similarity_matrix(self, similarity: Dict, output_dir: Path):
        """创建网络相似度矩阵图"""
        if not similarity:
            return
        
        regions = list(similarity.keys()) + [list(similarity[list(similarity.keys())[0]].keys())[0]] if similarity else []
        
        n = len(regions)
        matrix = np.zeros((n, n))
        
        for i, r1 in enumerate(regions):
            for j, r2 in enumerate(regions):
                if r1 in similarity and r2 in similarity[r1]:
                    matrix[i, j] = similarity[r1][r2]['overall_similarity']
                elif i == j:
                    matrix[i, j] = 1.0
                elif r2 in similarity and r1 in similarity.get(r2, {}):
                    matrix[i, j] = similarity[r2][r1]['overall_similarity']
        
        fig = go.Figure(data=go.Heatmap(
            z=matrix,
            x=regions,
            y=regions,
            colorscale='RdYlGn',
            zmin=0, zmax=1,
            hovertemplate='%{x} vs %{y}<br>相似度: %{z:.4f}<extra></extra>',
            text=[[f'{matrix[i][j]:.3f}' for j in range(n)] for i in range(n)],
            texttemplate='%{text}'
        ))
        
        fig.update_layout(
            title='网络整体相似度矩阵<br><sup>(Overall Network Similarity Matrix)</sup>',
            paper_bgcolor='#F5FAFF',
            font=dict(color='#1565C0')
        )
        
        output_path = output_dir / 'network_similarity_matrix.html'
        fig.write_html(str(output_path))
        logger.info(f"  ✓ 网络相似度矩阵: network_similarity_matrix.html")
    
    def _create_venn_diagram(self, node_overlap: Dict, output_dir: Path):
        """创建韦恩图展示共同/特有节点"""
        # 使用柱状图替代韦恩图（更清晰）
        common_all = node_overlap.get('common_to_all', [])
        unique_nodes = node_overlap.get('unique_nodes', {})
        
        if not unique_nodes:
            return
        
        regions = list(unique_nodes.keys())
        
        fig = go.Figure()
        
        # 共同节点
        fig.add_trace(go.Bar(
            name='三地区共同',
            x=regions,
            y=[len(common_all)] * len(regions),
            marker_color='#2E86AB'
        ))
        
        # 特有节点
        fig.add_trace(go.Bar(
            name='该地区特有',
            x=regions,
            y=[len(unique_nodes.get(r, [])) for r in regions],
            marker_color='#A23B72'
        ))
        
        fig.update_layout(
            title='各地区节点分布<br><sup>(Node Distribution by Region)</sup>',
            barmode='group',
            yaxis_title='节点数量',
            paper_bgcolor='#F5FAFF',
            font=dict(color='#1565C0')
        )
        
        output_path = output_dir / 'node_distribution.html'
        fig.write_html(str(output_path))
        logger.info(f"  ✓ 节点分布图: node_distribution.html")
    
    def generate_comparison_report(self, comparison_results: Dict, output_dir: Path):
        """生成对比分析报告"""
        logger.info("\n📊 阶段4: 生成对比分析报告")
        logger.info("─" * 60)
        
        report_path = output_dir / 'comparative_analysis_report.txt'
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("═" * 80 + "\n")
            f.write("中美欧语义网络对比分析报告\n")
            f.write("Comparative Semantic Network Analysis: China, US, EU\n")
            f.write("═" * 80 + "\n\n")
            
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # 1. 数据概览
            f.write("─" * 80 + "\n")
            f.write("一、数据概览\n")
            f.write("─" * 80 + "\n\n")
            
            for region, stats in self.region_stats.items():
                f.write(f"【{region}】\n")
                f.write(f"  节点数: {stats.get('n_nodes', 0)}\n")
                f.write(f"  边数: {stats.get('n_edges', 0)}\n")
                f.write(f"  网络密度: {stats.get('density', 0):.4f}\n")
                f.write(f"  平均聚类系数: {stats.get('avg_clustering', 0):.4f}\n")
                f.write(f"  社区数: {stats.get('n_communities', 0)}\n")
                f.write(f"  模块度: {stats.get('modularity', 0):.4f}\n\n")
            
            # 2. 网络拓扑对比
            f.write("─" * 80 + "\n")
            f.write("二、网络拓扑指标对比\n")
            f.write("─" * 80 + "\n\n")
            
            topology = comparison_results.get('topology_comparison', {})
            df = pd.DataFrame(topology.get('dataframe', []))
            if not df.empty:
                f.write(df.to_string(index=False))
                f.write("\n\n")
            
            # 3. 节点重叠分析
            f.write("─" * 80 + "\n")
            f.write("三、节点重叠分析\n")
            f.write("─" * 80 + "\n\n")
            
            node_overlap = comparison_results.get('node_overlap', {})
            f.write(f"三地区共同节点数: {len(node_overlap.get('common_to_all', []))}\n\n")
            
            for r1, overlaps in node_overlap.get('pairwise_overlap', {}).items():
                for r2, data in overlaps.items():
                    f.write(f"【{r1} vs {r2}】\n")
                    f.write(f"  Jaccard相似度: {data['jaccard_similarity']:.4f}\n")
                    f.write(f"  共同节点: {data['intersection_size']} 个\n")
                    f.write(f"  {r1}特有: {data['r1_unique']} 个\n")
                    f.write(f"  {r2}特有: {data['r2_unique']} 个\n\n")
            
            # 4. 核心概念对比
            f.write("─" * 80 + "\n")
            f.write("四、核心概念对比\n")
            f.write("─" * 80 + "\n\n")
            
            core_concepts = comparison_results.get('core_concepts_comparison', {})
            for region, data in core_concepts.items():
                f.write(f"【{region}】Top 10核心概念:\n")
                for node, score in data.get('top_by_degree', [])[:10]:
                    f.write(f"  - {node}: {score:.4f}\n")
                if data.get('unique_core'):
                    f.write(f"  特有核心概念: {', '.join(data['unique_core'][:10])}\n")
                f.write("\n")
            
            # 5. 网络相似度
            f.write("─" * 80 + "\n")
            f.write("五、网络整体相似度\n")
            f.write("─" * 80 + "\n\n")
            
            similarity = comparison_results.get('network_similarity', {})
            for r1, sims in similarity.items():
                for r2, data in sims.items():
                    f.write(f"【{r1} vs {r2}】\n")
                    f.write(f"  节点Jaccard相似度: {data['node_jaccard']:.4f}\n")
                    f.write(f"  度分布相关性: {data['degree_correlation']:.4f}\n")
                    f.write(f"  密度相似度: {data['density_similarity']:.4f}\n")
                    f.write(f"  综合相似度: {data['overall_similarity']:.4f}\n\n")
            
            # 6. 主要发现
            f.write("─" * 80 + "\n")
            f.write("六、主要发现与洞察\n")
            f.write("─" * 80 + "\n\n")
            
            # 自动生成洞察
            common_nodes = node_overlap.get('common_to_all', [])
            if common_nodes:
                f.write(f"1. 共同关注领域: 中美欧三地共同关注的概念有{len(common_nodes)}个，")
                f.write(f"主要包括: {', '.join(common_nodes[:15])}\n\n")
            
            unique = {r: len(node_overlap.get('unique_nodes', {}).get(r, [])) for r in self.region_networks.keys()}
            if unique:
                max_unique = max(unique, key=unique.get)
                f.write(f"2. 特色关注领域: {max_unique}有最多特有概念({unique[max_unique]}个)，")
                f.write(f"反映了该地区独特的政策关注点\n\n")
            
            f.write("═" * 80 + "\n")
            f.write("报告生成完成\n")
            f.write("═" * 80 + "\n")
        
        logger.info(f"  ✓ 对比报告: comparative_analysis_report.txt")
        
        # 保存JSON结果
        json_path = output_dir / 'comparative_analysis_results.json'
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(comparison_results, f, ensure_ascii=False, indent=2, default=str)
        logger.info(f"  ✓ JSON结果: comparative_analysis_results.json")


def run_china_us_eu_comparative_analysis():
    """运行中美欧对比分析
    
    分析流程:
    1. 加载所有数据
    2. 按国家筛选分为美国、中国、欧盟三组
    3. 分别构建语义网络
    4. 进行网络对比分析
    5. 生成对比可视化
    6. 输出对比报告
    """
    global progress
    
    logger.info("═" * 80)
    logger.info("🌍 中美欧语义网络对比分析系统启动")
    logger.info(f"📅 开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("═" * 80)
    
    # ========== 阶段1: 初始化 ==========
    progress.start_stage(0)
    config = NarrativeAnalysisConfig()
    progress.end_stage()
    
    # ========== 阶段2: 加载停用词 ==========
    progress.start_stage(1)
    stopwords_loader = StopwordsLoader(config.stopwords_paths)
    stop_words = stopwords_loader.load_all_stopwords()
    progress.end_stage()
    
    # ========== 阶段3: 加载元数据 ==========
    progress.start_stage(2)
    metadata_loader = MetadataLoader(config.metadata_dir)
    available_countries = metadata_loader.get_all_countries()
    logger.info(f"  可用国家/地区: {available_countries}")
    progress.end_stage()
    
    # ========== 阶段4: 加载文档 ==========
    progress.start_stage(3)
    doc_loader = DocumentLoader(
        config.agora_fulltext_dir,
        config.original_data_dir,
        metadata_loader
    )
    documents = doc_loader.load_documents()
    
    if not documents:
        logger.error("❌ 没有找到有效文档")
        return
    progress.end_stage()
    
    # ========== 阶段5: 按国家分组 ==========
    logger.info("\n📊 阶段5: 按国家/地区分组")
    logger.info("─" * 60)
    
    # 定义国家映射（处理可能的变体）
    country_mapping = {
        '美国': ['美国', 'US', 'USA', 'United States', 'United States of America'],
        '中国': ['中国', 'CN', 'China', 'PRC', '中华人民共和国'],
        '欧盟': ['欧盟', 'EU', 'European Union', '欧洲联盟', 'Europe']
    }
    
    region_docs = {'美国': [], '中国': [], '欧盟': []}
    
    for doc in documents:
        country = metadata_loader.get_country(doc['filename'], doc['data_source'])
        
        # 匹配到对应地区
        for region, aliases in country_mapping.items():
            if country in aliases or country == region:
                region_docs[region].append(doc)
                break
    
    # 显示各区域文档数
    for region, docs in region_docs.items():
        logger.info(f"  {region}: {len(docs)} 个文档")
    
    # 检查是否有足够数据
    valid_regions = {r: docs for r, docs in region_docs.items() if len(docs) >= 5}
    if len(valid_regions) < 2:
        logger.error("❌ 至少需要2个地区各有至少5个文档才能进行对比分析")
        logger.info(f"💡 当前数据: {[(r, len(d)) for r, d in region_docs.items()]}")
        return
    
    # ========== 阶段6-9: 对比分析 ==========
    # 创建对比分析器
    comparative_analyzer = ComparativeNetworkAnalyzer(config)
    comparative_analyzer.set_stop_words(stop_words)
    
    # 构建各地区网络
    region_networks = comparative_analyzer.build_regional_networks(valid_regions)
    
    if len(region_networks) < 2:
        logger.error("❌ 网络构建失败，无法进行对比")
        return
    
    # 进行网络对比
    comparison_results = comparative_analyzer.compare_networks()
    
    # 创建输出目录
    output_dir = config.output_dir / 'comparative_analysis'
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # 生成可视化
    comparative_analyzer.create_comparison_visualizations(comparison_results, output_dir)
    
    # 导出各网络到Gephi
    gephi_dir = output_dir / 'gephi'
    gephi_dir.mkdir(exist_ok=True)
    
    for region, G in region_networks.items():
        nx.write_gexf(G, str(gephi_dir / f'{region}_network.gexf'))
        nx.write_graphml(G, str(gephi_dir / f'{region}_network.graphml'))
    
    logger.info(f"\n  ✓ Gephi文件已导出至: {gephi_dir}")
    
    # 生成对比报告
    comparative_analyzer.generate_comparison_report(comparison_results, output_dir)
    
    # 完成
    progress.finish()
    
    logger.info(f"\n📁 对比分析输出目录: {output_dir}")
    logger.info("═" * 80)


if __name__ == "__main__":
    # 默认运行完整分析（不筛选）
    # run_full_analysis()
    
    # ============ 筛选示例 ============
    # 示例1: 只分析美国的文档
    # run_analysis_with_country_filter(['美国'])
    
    # 示例2: 分析美国和中国的文档
    run_analysis_with_country_filter(['美国', '中国'])
    
    # 示例3: 只分析政府发布的文档
    run_analysis_with_org_type_filter(['government'])
    
    # 示例4: 查看所有可用的筛选选项
    # list_available_filters()
    
    # 示例5: 中美欧对比分析
    # run_china_us_eu_comparative_analysis()
    
    # 默认运行完整分析
    run_full_analysis()
