# 文本语言学特征分析系统
# Text Linguistic Feature Analysis for Policy Documents
# 分析词汇丰富度、词汇难度、句法复杂度、信息密度与可预测性、结构与连贯性

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
import hashlib
import pickle
import shutil
import math
from scipy import stats
from scipy.spatial.distance import cosine

# NLP相关导入
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy

# 进度条
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

# Plotly可视化
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import plotly.figure_factory as ff
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    go = None
    px = None
    make_subplots = None
    ff = None


# ==================== 静默进度迭代器 ====================
class SilentProgress:
    """静默进度迭代器"""
    
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
            if self.current - self.last_report >= self.report_interval or self.current == self.total:
                pct = (self.current / self.total) * 100
                self.logger_func(f"    └─ {self.desc}: {self.current}/{self.total} ({pct:.1f}%)")
                self.last_report = self.current
            yield item
    
    def __len__(self):
        return self.total


# ==================== 进度追踪器 ====================
class ProgressTracker:
    """进度追踪器"""

    STAGES = [
        ("初始化配置", "⚙️"),
        ("加载停用词", "📚"),
        ("加载元数据", "📋"),
        ("加载文档", "📄"),
        ("词汇丰富度分析", "📊"),
        ("词汇难度分析", "📖"),
        ("句法复杂度分析", "🔤"),
        ("信息密度分析", "💡"),
        ("结构连贯性分析", "🔗"),
        ("保存结果", "💾"),
        ("生成报告", "📈")
    ]

    def __init__(self):
        self.current_stage = 0
        self.stage_start_time = time.time()
        self.total_start_time = time.time()
        self.stage_times = {}

    def start_stage(self, stage_idx: int = None):
        if stage_idx is not None:
            self.current_stage = stage_idx

        if self.current_stage > 0:
            elapsed = time.time() - self.stage_start_time
            prev_stage = self.STAGES[self.current_stage - 1][0]
            self.stage_times[prev_stage] = elapsed

        self.stage_start_time = time.time()

        if self.current_stage < len(self.STAGES):
            name, icon = self.STAGES[self.current_stage]
            progress = f"[{self.current_stage + 1}/{len(self.STAGES)}]"
            logger.info(f"\n{'─'*60}")
            logger.info(f"{icon} {progress} {name}")
            logger.info(f"{'─'*60}")

    def log_subtask(self, message: str, current: int = None, total: int = None):
        if current is not None and total is not None:
            pct = (current / total) * 100
            logger.info(f"  → {message} [{current}/{total}] ({pct:.1f}%)")
        else:
            logger.info(f"  → {message}")

    def log_metric(self, name: str, value):
        logger.info(f"  ✓ {name}: {value}")

    def end_stage(self):
        self.current_stage += 1

    def finish(self):
        total_time = time.time() - self.total_start_time

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
log_file = Path(__file__).parent / 'linguistic_analysis.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_file, encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)


# ==================== 配置类（复用现有框架）====================
class LinguisticAnalysisConfig:
    """文本语言学特征分析配置
    
    复用现有叙事分析系统的配置框架，保持一致性
    """
    
    def __init__(self):
        self._init_paths()
        self._init_filters()
        self._init_linguistic_settings()
        self._setup_output_dirs()
        
        logger.info(f"📁 输出目录: {self.output_dir}")
        if self.filter_countries:
            logger.info(f"🌍 国家筛选: {self.filter_countries}")
        if self.filter_org_types:
            logger.info(f"🏢 组织类型筛选: {self.filter_org_types}")
    
    def _init_paths(self):
        """初始化路径配置（与叙事分析系统一致）"""
        # 元数据路径
        self.metadata_dir = Path("/Volumes/ZimingYe/A_project/12月数据采集汇总/数据标注/1226标注结果")
        
        # 数据源路径
        self.agora_fulltext_dir = Path("/Volumes/ZimingYe/A_project/12月数据采集汇总/translated_data/fulltext")
        self.original_data_dir = Path("/Volumes/ZimingYe/A_project/12月数据采集汇总/translated_data/DATA")
        
        # 输出目录
        self.base_output_dir = Path("/Volumes/ZimingYe/A_project/Agroa数据汇总分析/output")
        
        # 缓存目录
        self.cache_dir = Path("/Volumes/ZimingYe/A_project/Agroa数据汇总分析/.cache")
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        
        # 停用词路径
        self.stopwords_paths = [
            "/Users/ziming_ye/Downloads/stopwords-iso.json",
            "/Users/ziming_ye/Python/hit_stopwords.txt",
            "/Users/ziming_ye/Python/cn_all_stopwords.txt"
        ]
        
        # 词汇难度参考词表路径
        self.word_freq_path = Path("/Volumes/ZimingYe/A_project/Agroa数据汇总分析/data/word_frequency.json")
        self.policy_terms_path = Path("/Volumes/ZimingYe/A_project/Agroa数据汇总分析/data/policy_terms.json")
    
    def _init_filters(self):
        """初始化数据筛选配置（与叙事分析系统一致）"""
        self.min_valid_year = 2015
        self.max_valid_year = 2025
        
        # 筛选条件
        self.filter_countries = []  # 空列表表示不筛选
        self.filter_org_types = []
        self.filter_doc_types = []
        self.filter_data_sources = []
    
    def _init_linguistic_settings(self):
        """初始化语言学分析参数"""
        # 文本处理参数
        self.max_doc_length = 5000000
        self.min_doc_length = 100
        self.min_sentence_length = 5  # 最小句子长度
        
        # 词汇丰富度参数
        self.ttr_sample_sizes = [100, 200, 500, 1000]  # TTR采样的文本长度
        
        # 词汇难度参数
        self.low_freq_threshold = 0.001  # 低频词阈值
        self.academic_word_ratio_threshold = 0.1  # 学术词汇比例阈值
        
        # 句法复杂度参数
        self.max_parse_depth = 10  # 最大句法分析深度
        
        # 信息密度参数
        self.ngram_range = (1, 3)  # n-gram范围
        
        # Spacy模型
        self.spacy_model_en = 'en_core_web_sm'
        self.spacy_model_zh = 'zh_core_web_sm'
    
    def _setup_output_dirs(self):
        """创建输出目录结构"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        filter_suffix = ""
        if self.filter_countries:
            filter_suffix += f"_countries_{'-'.join(self.filter_countries[:3])}"
            if len(self.filter_countries) > 3:
                filter_suffix += f"_{len(self.filter_countries)}"
        if self.filter_org_types:
            filter_suffix += f"_orgtypes_{len(self.filter_org_types)}"
        
        self.output_dir = self.base_output_dir / f"linguistic_analysis_{timestamp}{filter_suffix}"
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # 子目录
        for subdir in ['visualizations', 'reports', 'data', 'comparisons']:
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
        """设置国家筛选"""
        self.filter_countries = countries
        logger.info(f"🌍 已设置国家筛选: {countries}")
    
    def set_org_type_filter(self, org_types: List[str]):
        """设置组织类型筛选"""
        self.filter_org_types = org_types
        logger.info(f"🏢 已设置组织类型筛选: {org_types}")
    
    def clear_filters(self):
        """清除所有筛选条件"""
        self.filter_countries = []
        self.filter_org_types = []
        self.filter_doc_types = []
        self.filter_data_sources = []
        logger.info("🧹 已清除所有筛选条件")


# ==================== 停用词加载器 ====================
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
            'article', 'section', 'chapter', 'paragraph', 'clause'
        }
        self.stop_words.update(policy_stopwords)
        
        # 添加数字作为停用词
        for i in range(100000):
            self.stop_words.add(str(i))
        
        logger.info(f"📚 已加载 {len(self.stop_words)} 个停用词")
        return self.stop_words


# ==================== 元数据加载器 ====================
class MetadataLoader:
    """元数据加载器 - 支持多字段筛选"""
    
    def __init__(self, metadata_dir: Path):
        self.metadata_dir = metadata_dir
        self.metadata_cache = {}
        self._load_metadata()
    
    def _load_metadata(self):
        """加载所有元数据文件"""
        json_files = [f for f in self.metadata_dir.glob("*.json") if not f.name.startswith('._')]
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, dict) and 'files' in data:
                        items = data['files']
                    elif isinstance(data, list):
                        items = data
                    else:
                        items = []
                    
                    for item in items:
                        if 'filename' in item:
                            filename = item['filename']
                            if filename.endswith('.txt'):
                                filename = filename[:-4]
                            self.metadata_cache[filename] = item
            except Exception as e:
                logger.warning(f"⚠️ 加载元数据文件失败 {json_file}: {e}")
        
        logger.info(f"📚 已加载 {len(self.metadata_cache)} 条元数据")
    
    def get_metadata(self, filename: str) -> Dict:
        """获取文档元数据"""
        filename_clean = filename.replace('.txt', '')
        return self.metadata_cache.get(filename_clean, {})
    
    def get_year(self, filename: str, data_source: str) -> int:
        """获取文档年份"""
        metadata = self.get_metadata(filename)
        year = metadata.get('year') or metadata.get('publication_year') or metadata.get('Year')
        return year if year else 2020
    
    def get_org_type(self, filename: str, data_source: str) -> str:
        """获取组织类型"""
        metadata = self.get_metadata(filename)
        return metadata.get('level2_org_type') or metadata.get('org_type') or 'unknown'
    
    def get_country(self, filename: str, data_source: str) -> str:
        """获取国家/地区"""
        metadata = self.get_metadata(filename)
        return metadata.get('level1_country_or_org') or metadata.get('country') or 'unknown'
    
    def get_doc_type(self, filename: str, data_source: str) -> str:
        """获取文档类型"""
        metadata = self.get_metadata(filename)
        return metadata.get('level3_doc_type') or metadata.get('doc_type') or 'unknown'
    
    def get_organization(self, filename: str, data_source: str) -> str:
        """获取组织名称"""
        metadata = self.get_metadata(filename)
        return metadata.get('organization') or metadata.get('Organization') or 'unknown'


# ==================== 文档加载器 ====================
class DocumentLoader:
    """文档加载器"""
    
    def __init__(self, agora_dir: Path, data_dir: Path, metadata_loader: MetadataLoader, config: LinguisticAnalysisConfig):
        self.agora_dir = agora_dir
        self.data_dir = data_dir
        self.metadata_loader = metadata_loader
        self.config = config
    
    def load_documents(self) -> List[Dict]:
        """加载所有文档并应用筛选条件"""
        documents = []
        
        # 加载fulltext文件夹
        if self.agora_dir.exists():
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
        
        # 应用筛选条件
        documents = self._apply_filters(documents)
        
        logger.info(f"📄 已加载 {len(documents)} 个文档（筛选后）")
        return documents
    
    def _apply_filters(self, documents: List[Dict]) -> List[Dict]:
        """应用筛选条件"""
        filtered = []
        
        for doc in documents:
            filename = doc['filename']
            data_source = doc['data_source']
            
            # 国家筛选
            if self.config.filter_countries:
                country = self.metadata_loader.get_country(filename, data_source)
                if country not in self.config.filter_countries:
                    continue
            
            # 组织类型筛选
            if self.config.filter_org_types:
                org_type = self.metadata_loader.get_org_type(filename, data_source)
                if org_type not in self.config.filter_org_types:
                    continue
            
            # 文档类型筛选
            if self.config.filter_doc_types:
                doc_type = self.metadata_loader.get_doc_type(filename, data_source)
                if doc_type not in self.config.filter_doc_types:
                    continue
            
            # 数据源筛选
            if self.config.filter_data_sources:
                if data_source not in self.config.filter_data_sources:
                    continue
            
            filtered.append(doc)
        
        return filtered


# ==================== 词汇丰富度分析器 ====================
class VocabularyRichnessAnalyzer:
    """词汇丰富度分析器 - TTR改进版
    
    分析指标：
    1. TTR (Type-Token Ratio): 词汇类型/总词数
    2. RTTR (Root TTR): TTR * sqrt(N)
    3. CTTR (Corrected TTR): TTR * sqrt(2N)
    4. Herdan's C: log(V) / log(N)
    5. Maas's Index: (log(N) - log(V)) / log²(N)
    6. Honoré's Statistic: 100 * log(N) / (1 - V1/V)
    7. Brunet's Index: (V - N^a) / N, a ≈ -0.165
    8. Yule's K: 衡量词汇重复度
    9. Simpson's D: 词汇多样性
    """
    
    def __init__(self, config: LinguisticAnalysisConfig, stop_words: Set[str]):
        self.config = config
        self.stop_words = stop_words
        self.nlp_en = None
        self.nlp_zh = None
        self.cache_manager = None  # 缓存管理器
    
    def _load_spacy(self, lang: str = 'en'):
        """加载Spacy模型"""
        if lang == 'en' and self.nlp_en is None:
            try:
                self.nlp_en = spacy.load(self.config.spacy_model_en, disable=['ner', 'textcat'])
            except OSError:
                logger.warning(f"⚠️ 未找到Spacy模型 {self.config.spacy_model_en}，跳过句法分析")
        elif lang == 'zh' and self.nlp_zh is None:
            try:
                self.nlp_zh = spacy.load(self.config.spacy_model_zh, disable=['ner', 'textcat'])
            except OSError:
                logger.warning(f"⚠️ 未找到Spacy模型 {self.config.spacy_model_zh}，跳过句法分析")
    
    def _tokenize(self, text: str, lang: str = 'en') -> List[str]:
        """分词"""
        if lang == 'zh':
            # 中文简单分词
            words = list(text)
            words = [w for w in words if w.strip() and not w.isspace()]
        else:
            # 英文分词
            words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
        
        # 过滤停用词
        words = [w for w in words if w not in self.stop_words and len(w) > 1]
        return words
    
    def _detect_language(self, text: str) -> str:
        """检测文本语言"""
        # 简单的启发式检测
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        total_chars = len(text.replace(' ', ''))
        
        if total_chars > 0 and chinese_chars / total_chars > 0.3:
            return 'zh'
        return 'en'
    
    def analyze_document(self, text: str) -> Dict:
        """分析单个文档的词汇丰富度"""
        lang = self._detect_language(text[:1000])
        words = self._tokenize(text, lang)
        
        if len(words) < 50:
            return {'error': '文本过短', 'word_count': len(words)}
        
        # 基本统计
        n = len(words)  # 总词数 (tokens)
        v = len(set(words))  # 词汇类型数 (types)
        
        # 词频分布
        word_freq = Counter(words)
        freq_distribution = Counter(word_freq.values())
        
        results = {
            'language': lang,
            'word_count': n,
            'vocabulary_size': v,
            'unique_ratio': v / n if n > 0 else 0,
        }
        
        # 1. TTR (Type-Token Ratio)
        ttr = v / n if n > 0 else 0
        results['ttr'] = ttr
        
        # 2. RTTR (Root TTR)
        rttr = ttr * math.sqrt(n) if n > 0 else 0
        results['rttr'] = rttr
        
        # 3. CTTR (Corrected TTR)
        cttr = ttr * math.sqrt(2 * n) if n > 0 else 0
        results['cttr'] = cttr
        
        # 4. Herdan's C
        if n > 1 and v > 0:
            herdan_c = math.log(v) / math.log(n)
        else:
            herdan_c = 0
        results['herdan_c'] = herdan_c
        
        # 5. Maas's Index
        if n > 1 and v > 0:
            log_n = math.log(n)
            log_v = math.log(v)
            maas = (log_n - log_v) / (log_n ** 2)
        else:
            maas = 0
        results['maas_index'] = maas
        
        # 6. Honoré's Statistic
        v1 = freq_distribution.get(1, 0)  # 仅出现一次的词数
        if n > 0 and v > 0 and v != v1:
            honore = 100 * math.log(n) / (1 - v1 / v)
        else:
            honore = 0
        results['honore_stat'] = honore
        
        # 7. Brunet's Index
        a = -0.165  # 经验常数
        brunet = n ** a if n > 0 else 0
        results['brunet_index'] = brunet
        
        # 8. Yule's K
        if n > 1:
            m2 = sum(f * f * count for f, count in freq_distribution.items())
            yule_k = (m2 - n) / (n * n) if n > 0 else 0
        else:
            yule_k = 0
        results['yule_k'] = yule_k
        
        # 9. Simpson's D
        if n > 1:
            simpson_d = sum(f * (f - 1) * count for f, count in freq_distribution.items()) / (n * (n - 1))
        else:
            simpson_d = 0
        results['simpson_d'] = simpson_d
        
        # 10. 词汇密度指标
        results['lexical_density'] = v / n if n > 0 else 0
        
        # 11. hapax_legomena_ratio (仅出现一次的词比例)
        results['hapax_legomena_ratio'] = v1 / v if v > 0 else 0
        
        # 12. dislegomena_ratio (仅出现两次的词比例)
        v2 = freq_distribution.get(2, 0)
        results['dislegomena_ratio'] = v2 / v if v > 0 else 0
        
        return results
    
    def analyze_batch(self, documents: List[Dict]) -> Dict:
        """批量分析文档（支持缓存）"""
        results = {}
        
        for doc in SilentProgress(documents, desc="词汇丰富度分析", report_interval=50):
            doc_id = doc['filename']
            
            # 尝试从缓存获取
            if self.cache_manager:
                cached = self.cache_manager.get_cached_result(doc_id, 'richness')
                if cached:
                    results[doc_id] = cached
                    results[doc_id]['data_source'] = doc['data_source']
                    continue
            
            # 分析文档
            result = self.analyze_document(doc['content'])
            result['data_source'] = doc['data_source']
            results[doc_id] = result
            
            # 保存到缓存
            if self.cache_manager and 'error' not in result:
                self.cache_manager.save_result(doc_id, 'richness', result)
        
        return results
    
    def get_summary_statistics(self, results: Dict) -> Dict:
        """获取汇总统计"""
        metrics = ['ttr', 'rttr', 'cttr', 'herdan_c', 'maas_index', 
                   'honore_stat', 'brunet_index', 'yule_k', 'simpson_d',
                   'lexical_density', 'hapax_legomena_ratio']
        
        summary = {}
        for metric in metrics:
            values = [r.get(metric, 0) for r in results.values() if 'error' not in r]
            if values:
                summary[metric] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'median': np.median(values),
                    'min': np.min(values),
                    'max': np.max(values)
                }
        
        return summary


# ==================== 词汇难度分析器 ====================
class VocabularyDifficultyAnalyzer:
    """词汇难度分析器 - 针对治理文本的评估指标
    
    分析指标：
    1. 平均词长
    2. 长词比例 (>6字母)
    3. 低频词比例
    4. 学术/专业词汇比例
    5. 政策术语密度
    6. 词汇复杂度指数
    7. 抽象名词比例
    8. 法律术语密度
    """
    
    def __init__(self, config: LinguisticAnalysisConfig, stop_words: Set[str]):
        self.config = config
        self.stop_words = stop_words
        self.cache_manager = None  # 缓存管理器
        
        # 学术词汇表（示例）
        self.academic_words = self._load_academic_words()
        
        # 政策术语表
        self.policy_terms = self._load_policy_terms()
        
        # 法律术语表
        self.legal_terms = self._load_legal_terms()
        
        # 抽象名词后缀
        self.abstract_noun_suffixes = ['tion', 'sion', 'ment', 'ness', 'ity', 
                                        'ance', 'ence', 'ism', 'ship', 'dom',
                                        'ure', 'age', 'ery', 'ice', 'ity']
    
    def _load_academic_words(self) -> Set[str]:
        """加载学术词汇表"""
        # 学术英语常用词汇（简化版）
        academic_words = {
            'analysis', 'approach', 'assessment', 'assume', 'authority',
            'available', 'benefit', 'category', 'challenge', 'concept',
            'conclusion', 'conduct', 'consequence', 'construct', 'context',
            'contribute', 'criteria', 'define', 'demonstrate', 'derive',
            'determine', 'document', 'dominant', 'effect', 'element',
            'emphasize', 'ensure', 'environment', 'establish', 'estimate',
            'evaluate', 'evidence', 'factor', 'framework', 'function',
            'fundamental', 'hypothesis', 'identify', 'implement', 'implication',
            'indicate', 'individual', 'interpret', 'investigate', 'issue',
            'method', 'objective', 'obvious', 'participant', 'perspective',
            'potential', 'previous', 'primary', 'principle', 'proceed',
            'process', 'prominent', 'proportion', 'protocol', 'relevant',
            'report', 'require', 'research', 'respond', 'role',
            'scope', 'section', 'significant', 'similar', 'source',
            'specific', 'strategy', 'structure', 'subsequent', 'substantial',
            'theoretical', 'thesis', 'topic', 'transform', 'trend',
            'variable', 'variant', 'variation', 'verify', 'volume',
            # 添加更多学术词汇
            'paradigm', 'phenomenon', 'methodology', 'infrastructure',
            'implementation', 'stakeholder', 'governance', 'mechanism',
            'framework', 'comprehensive', 'integration', 'optimization'
        }
        return academic_words
    
    def _load_policy_terms(self) -> Set[str]:
        """加载政策术语表"""
        policy_terms = {
            'regulation', 'legislation', 'policy', 'compliance', 'enforcement',
            'mandate', 'directive', 'ordinance', 'statute', 'decree',
            'jurisdiction', 'jurisdictional', 'jurisprudence', 'legislative',
            'executive', 'administrative', 'constitutional', 'provision',
            'amendment', 'ratification', 'implementation', 'enactment',
            'oversight', 'accountability', 'transparency', 'governance',
            'stakeholder', 'policymaker', 'legislature', 'bureaucracy',
            'deregulation', 'privatization', 'liberalization', 'reform',
            'initiative', 'framework', 'protocol', 'guideline', 'standard',
            'certification', 'accreditation', 'authorization', 'licensing',
            'incentive', 'subsidy', 'taxation', 'appropriation', 'allocation',
            # AI治理相关
            'algorithm', 'artificial', 'intelligence', 'machine', 'learning',
            'neural', 'network', 'automation', 'autonomous', 'ethical',
            'bias', 'fairness', 'transparency', 'accountability', 'privacy',
            'data', 'protection', 'security', 'cybersecurity', 'digital'
        }
        return policy_terms
    
    def _load_legal_terms(self) -> Set[str]:
        """加载法律术语表"""
        legal_terms = {
            'plaintiff', 'defendant', 'verdict', 'judgment', 'litigation',
            'jurisdiction', 'jurisprudence', 'tort', 'contract', 'liability',
            'negligence', 'damages', 'remedy', 'injunction', 'subpoena',
            'affidavit', 'deposition', 'testimony', 'evidence', 'witness',
            'counsel', 'attorney', 'prosecutor', 'defendant', 'appellant',
            'appellee', 'petitioner', 'respondent', 'arbitration', 'mediation',
            'settlement', 'indemnification', 'warranty', 'covenant', 'provision',
            'herein', 'thereof', 'whereas', 'hereto', 'therein', 'aforesaid',
            'notwithstanding', 'pursuant', 'hereby', 'thereto', 'whereof'
        }
        return legal_terms
    
    def _detect_language(self, text: str) -> str:
        """检测文本语言"""
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        total_chars = len(text.replace(' ', ''))
        if total_chars > 0 and chinese_chars / total_chars > 0.3:
            return 'zh'
        return 'en'
    
    def _tokenize(self, text: str, lang: str = 'en') -> List[str]:
        """分词"""
        if lang == 'zh':
            words = list(text)
            words = [w for w in words if w.strip() and not w.isspace()]
        else:
            words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
        return [w for w in words if w not in self.stop_words and len(w) > 1]
    
    def analyze_document(self, text: str) -> Dict:
        """分析单个文档的词汇难度"""
        lang = self._detect_language(text[:1000])
        words = self._tokenize(text, lang)
        
        if len(words) < 50:
            return {'error': '文本过短', 'word_count': len(words)}
        
        results = {
            'language': lang,
            'word_count': len(words)
        }
        
        # 1. 平均词长
        word_lengths = [len(w) for w in words]
        results['avg_word_length'] = np.mean(word_lengths)
        
        # 2. 长词比例 (>6字母)
        long_words = [w for w in words if len(w) > 6]
        results['long_word_ratio'] = len(long_words) / len(words) if words else 0
        
        # 3. 非常长词比例 (>10字母)
        very_long_words = [w for w in words if len(w) > 10]
        results['very_long_word_ratio'] = len(very_long_words) / len(words) if words else 0
        
        # 4. 学术词汇比例
        academic_count = sum(1 for w in words if w in self.academic_words)
        results['academic_word_ratio'] = academic_count / len(words) if words else 0
        
        # 5. 政策术语密度
        policy_count = sum(1 for w in words if w in self.policy_terms)
        results['policy_term_density'] = policy_count / len(words) if words else 0
        
        # 6. 法律术语密度
        legal_count = sum(1 for w in words if w in self.legal_terms)
        results['legal_term_density'] = legal_count / len(words) if words else 0
        
        # 7. 抽象名词比例（基于后缀）
        if lang == 'en':
            abstract_nouns = [w for w in words 
                            if any(w.endswith(suffix) for suffix in self.abstract_noun_suffixes)]
            results['abstract_noun_ratio'] = len(abstract_nouns) / len(words) if words else 0
        else:
            results['abstract_noun_ratio'] = 0
        
        # 8. 词汇复杂度指数（综合指标）
        complexity_index = (
            results['avg_word_length'] * 0.2 +
            results['long_word_ratio'] * 10 * 0.2 +
            results['academic_word_ratio'] * 20 * 0.2 +
            results['policy_term_density'] * 15 * 0.2 +
            results['legal_term_density'] * 15 * 0.2
        )
        results['vocabulary_complexity_index'] = complexity_index
        
        # 9. 高级词汇比例（综合学术+政策+法律）
        advanced_count = academic_count + policy_count + legal_count
        results['advanced_vocabulary_ratio'] = advanced_count / len(words) if words else 0
        
        # 10. 词汇多样性（去重后比例）
        unique_words = set(words)
        results['vocabulary_diversity'] = len(unique_words) / len(words) if words else 0
        
        return results
    
    def analyze_batch(self, documents: List[Dict]) -> Dict:
        """批量分析文档（支持缓存）"""
        results = {}
        
        for doc in SilentProgress(documents, desc="词汇难度分析", report_interval=50):
            doc_id = doc['filename']
            
            # 尝试从缓存获取
            if self.cache_manager:
                cached = self.cache_manager.get_cached_result(doc_id, 'difficulty')
                if cached:
                    results[doc_id] = cached
                    results[doc_id]['data_source'] = doc['data_source']
                    continue
            
            # 分析文档
            result = self.analyze_document(doc['content'])
            result['data_source'] = doc['data_source']
            results[doc_id] = result
            
            # 保存到缓存
            if self.cache_manager and 'error' not in result:
                self.cache_manager.save_result(doc_id, 'difficulty', result)
        
        return results
    
    def get_summary_statistics(self, results: Dict) -> Dict:
        """获取汇总统计"""
        metrics = ['avg_word_length', 'long_word_ratio', 'very_long_word_ratio',
                   'academic_word_ratio', 'policy_term_density', 'legal_term_density',
                   'abstract_noun_ratio', 'vocabulary_complexity_index', 
                   'advanced_vocabulary_ratio', 'vocabulary_diversity']
        
        summary = {}
        for metric in metrics:
            values = [r.get(metric, 0) for r in results.values() if 'error' not in r]
            if values:
                summary[metric] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'median': np.median(values),
                    'min': np.min(values),
                    'max': np.max(values)
                }
        
        return summary


# ==================== 句法复杂度分析器 ====================
class SyntacticComplexityAnalyzer:
    """句法复杂度分析器
    
    分析指标：
    1. 平均句长（词数）
    2. 平均句长（字符数）
    3. 句子长度变异性
    4. 从句比例
    5. 平均从句嵌套深度
    6. 句子结构多样性
    7. 复杂句比例
    8. 并列结构比例
    9. 依存句法平均深度
    10. 语法树平均深度
    """
    
    def __init__(self, config: LinguisticAnalysisConfig, stop_words: Set[str]):
        self.config = config
        self.stop_words = stop_words
        self.nlp_en = None
        self.nlp_zh = None
        self.cache_manager = None  # 缓存管理器
    
    def _load_spacy(self, lang: str = 'en'):
        """加载Spacy模型"""
        if lang == 'en' and self.nlp_en is None:
            try:
                self.nlp_en = spacy.load(self.config.spacy_model_en, disable=['ner', 'textcat'])
                logger.info("  ✓ 加载英文Spacy模型完成")
            except OSError:
                logger.warning(f"⚠️ 未找到Spacy模型 {self.config.spacy_model_en}")
        elif lang == 'zh' and self.nlp_zh is None:
            try:
                self.nlp_zh = spacy.load(self.config.spacy_model_zh, disable=['ner', 'textcat'])
                logger.info("  ✓ 加载中文Spacy模型完成")
            except OSError:
                logger.warning(f"⚠️ 未找到Spacy模型 {self.config.spacy_model_zh}")
    
    def _detect_language(self, text: str) -> str:
        """检测文本语言"""
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        total_chars = len(text.replace(' ', ''))
        if total_chars > 0 and chinese_chars / total_chars > 0.3:
            return 'zh'
        return 'en'
    
    def _split_sentences_simple(self, text: str, lang: str = 'en') -> List[str]:
        """简单句子分割（不依赖Spacy）"""
        if lang == 'zh':
            # 中文句子分割
            sentences = re.split(r'[。！？；\n]', text)
        else:
            # 英文句子分割
            sentences = re.split(r'(?<=[.!?])\s+', text)
        
        return [s.strip() for s in sentences if len(s.strip()) > 10]
    
    def _analyze_sentence_basic(self, sentence: str) -> Dict:
        """基本句子分析（不依赖Spacy）"""
        words = re.findall(r'\b[a-zA-Z]+\b', sentence.lower())
        words = [w for w in words if w not in self.stop_words]
        
        return {
            'word_count': len(words),
            'char_count': len(sentence),
            'has_comma': ',' in sentence,
            'has_semicolon': ';' in sentence,
            'has_colon': ':' in sentence,
            'has_dash': '-' in sentence or '—' in sentence,
            'has_parentheses': '(' in sentence,
            'punctuation_count': sum(1 for c in sentence if c in ',;:-—()')
        }
    
    def _analyze_sentence_spacy(self, doc, lang: str = 'en') -> Dict:
        """使用Spacy进行深度句法分析"""
        results = {}
        
        # 基本统计
        tokens = [t for t in doc if not t.is_punct and not t.is_space]
        results['word_count'] = len(tokens)
        results['char_count'] = len(doc.text)
        
        # 依存句法深度
        def get_depth(token, depth=0):
            max_depth = depth
            for child in token.children:
                max_depth = max(max_depth, get_depth(child, depth + 1))
            return max_depth
        
        # 找到根节点
        roots = [t for t in doc if t.head == t]
        if roots:
            results['parse_tree_depth'] = max(get_depth(root) for root in roots)
        else:
            results['parse_tree_depth'] = 0
        
        # 从句统计
        clause_markers = ['because', 'although', 'while', 'if', 'when', 'where', 
                         'that', 'which', 'who', 'whom', 'whose', 'since', 'unless']
        
        clauses = [t for t in doc if t.text.lower() in clause_markers]
        results['clause_count'] = len(clauses)
        
        # 从属连词
        subordinating_conjunctions = [t for t in doc if t.pos_ == 'SCONJ']
        results['subordinating_conjunction_count'] = len(subordinating_conjunctions)
        
        # 并列连词
        coordinating_conjunctions = [t for t in doc if t.pos_ == 'CCONJ']
        results['coordinating_conjunction_count'] = len(coordinating_conjunctions)
        
        # 词性分布
        pos_counts = Counter([t.pos_ for t in tokens])
        results['pos_distribution'] = dict(pos_counts)
        
        # 名词密度
        noun_count = pos_counts.get('NOUN', 0) + pos_counts.get('PROPN', 0)
        results['noun_density'] = noun_count / len(tokens) if tokens else 0
        
        # 动词密度
        verb_count = pos_counts.get('VERB', 0) + pos_counts.get('AUX', 0)
        results['verb_density'] = verb_count / len(tokens) if tokens else 0
        
        # 形容词密度
        adj_count = pos_counts.get('ADJ', 0)
        results['adjective_density'] = adj_count / len(tokens) if tokens else 0
        
        # 副词密度
        adv_count = pos_counts.get('ADV', 0)
        results['adverb_density'] = adv_count / len(tokens) if tokens else 0
        
        # 介词密度
        prep_count = pos_counts.get('ADP', 0)
        results['preposition_density'] = prep_count / len(tokens) if tokens else 0
        
        return results
    
    def analyze_document(self, text: str) -> Dict:
        """分析单个文档的句法复杂度"""
        lang = self._detect_language(text[:1000])
        
        # 分句
        sentences = self._split_sentences_simple(text, lang)
        
        if len(sentences) < 3:
            return {'error': '句子数量不足', 'sentence_count': len(sentences)}
        
        results = {
            'language': lang,
            'sentence_count': len(sentences)
        }
        
        # 基本句法分析（不依赖Spacy）
        sentence_lengths_words = []
        sentence_lengths_chars = []
        punctuation_counts = []
        complex_sentence_indicators = 0
        
        for sent in sentences:
            analysis = self._analyze_sentence_basic(sent)
            sentence_lengths_words.append(analysis['word_count'])
            sentence_lengths_chars.append(analysis['char_count'])
            punctuation_counts.append(analysis['punctuation_count'])
            
            # 复杂句指标
            if analysis['has_comma'] and analysis['punctuation_count'] >= 2:
                complex_sentence_indicators += 1
        
        # 1. 平均句长（词数）
        results['mean_sentence_length_words'] = np.mean(sentence_lengths_words)
        
        # 2. 平均句长（字符数）
        results['mean_sentence_length_chars'] = np.mean(sentence_lengths_chars)
        
        # 3. 句子长度变异性（标准差）
        results['sentence_length_variability'] = np.std(sentence_lengths_words)
        
        # 4. 句子长度变异系数
        mean_len = results['mean_sentence_length_words']
        results['sentence_length_cv'] = results['sentence_length_variability'] / mean_len if mean_len > 0 else 0
        
        # 5. 复杂句比例
        results['complex_sentence_ratio'] = complex_sentence_indicators / len(sentences) if sentences else 0
        
        # 6. 平均标点数
        results['mean_punctuation_per_sentence'] = np.mean(punctuation_counts)
        
        # 7. 长句比例（>25词）
        long_sentences = [l for l in sentence_lengths_words if l > 25]
        results['long_sentence_ratio'] = len(long_sentences) / len(sentences) if sentences else 0
        
        # 8. 短句比例（<10词）
        short_sentences = [l for l in sentence_lengths_words if l < 10]
        results['short_sentence_ratio'] = len(short_sentences) / len(sentences) if sentences else 0
        
        # 9. 句子长度分布
        results['sentence_length_distribution'] = {
            'q25': np.percentile(sentence_lengths_words, 25),
            'q50': np.percentile(sentence_lengths_words, 50),
            'q75': np.percentile(sentence_lengths_words, 75)
        }
        
        # 10. 句法复杂度综合指数
        complexity_index = (
            (results['mean_sentence_length_words'] / 20) * 0.2 +
            results['complex_sentence_ratio'] * 0.3 +
            (results['mean_punctuation_per_sentence'] / 3) * 0.2 +
            results['long_sentence_ratio'] * 0.3
        )
        results['syntactic_complexity_index'] = complexity_index
        
        # 尝试使用Spacy进行更深度分析
        try:
            self._load_spacy(lang)
            nlp = self.nlp_en if lang == 'en' else self.nlp_zh
            
            if nlp:
                spacy_results = self._analyze_with_spacy(text, nlp, lang)
                results.update(spacy_results)
        except Exception as e:
            logger.debug(f"Spacy分析跳过: {e}")
        
        return results
    
    def _analyze_with_spacy(self, text: str, nlp, lang: str) -> Dict:
        """使用Spacy进行深度分析"""
        # 限制文本长度避免内存问题
        text = text[:50000]
        
        doc = nlp(text)
        sentences = list(doc.sents)
        
        if not sentences:
            return {}
        
        results = {}
        
        # 依存句法深度统计
        parse_depths = []
        clause_counts = []
        
        for sent in sentences:
            sent_analysis = self._analyze_sentence_spacy(sent, lang)
            parse_depths.append(sent_analysis.get('parse_tree_depth', 0))
            clause_counts.append(sent_analysis.get('clause_count', 0))
        
        # 平均依存深度
        results['mean_parse_tree_depth'] = np.mean(parse_depths) if parse_depths else 0
        
        # 最大依存深度
        results['max_parse_tree_depth'] = max(parse_depths) if parse_depths else 0
        
        # 平均从句数
        results['mean_clause_count'] = np.mean(clause_counts) if clause_counts else 0
        
        # 从句比例
        sentences_with_clauses = sum(1 for c in clause_counts if c > 0)
        results['clause_sentence_ratio'] = sentences_with_clauses / len(sentences) if sentences else 0
        
        return results
    
    def analyze_batch(self, documents: List[Dict]) -> Dict:
        """批量分析文档（支持缓存）"""
        results = {}
        
        for doc in SilentProgress(documents, desc="句法复杂度分析", report_interval=50):
            doc_id = doc['filename']
            
            # 尝试从缓存获取
            if self.cache_manager:
                cached = self.cache_manager.get_cached_result(doc_id, 'syntactic')
                if cached:
                    results[doc_id] = cached
                    results[doc_id]['data_source'] = doc['data_source']
                    continue
            
            # 分析文档
            result = self.analyze_document(doc['content'])
            result['data_source'] = doc['data_source']
            results[doc_id] = result
            
            # 保存到缓存
            if self.cache_manager and 'error' not in result:
                self.cache_manager.save_result(doc_id, 'syntactic', result)
        
        return results
    
    def get_summary_statistics(self, results: Dict) -> Dict:
        """获取汇总统计"""
        metrics = ['mean_sentence_length_words', 'mean_sentence_length_chars',
                   'sentence_length_variability', 'sentence_length_cv',
                   'complex_sentence_ratio', 'mean_punctuation_per_sentence',
                   'long_sentence_ratio', 'short_sentence_ratio',
                   'syntactic_complexity_index', 'mean_parse_tree_depth',
                   'mean_clause_count', 'clause_sentence_ratio']
        
        summary = {}
        for metric in metrics:
            values = [r.get(metric, 0) for r in results.values() if 'error' not in r and metric in r]
            if values:
                summary[metric] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'median': np.median(values),
                    'min': np.min(values),
                    'max': np.max(values)
                }
        
        return summary


# ==================== 信息密度分析器 ====================
class InformationDensityAnalyzer:
    """信息密度分析器
    
    分析指标：
    1. 信息熵
    2. 困惑度估算
    3. 词汇重复率
    4. n-gram多样性
    5. 内容词密度
    6. 信息密度指数
    7. 新信息比例
    8. 词汇冗余度
    """
    
    def __init__(self, config: LinguisticAnalysisConfig, stop_words: Set[str]):
        self.config = config
        self.stop_words = stop_words
        self.cache_manager = None  # 缓存管理器
    
    def _detect_language(self, text: str) -> str:
        """检测文本语言"""
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        total_chars = len(text.replace(' ', ''))
        if total_chars > 0 and chinese_chars / total_chars > 0.3:
            return 'zh'
        return 'en'
    
    def _tokenize(self, text: str, lang: str = 'en') -> List[str]:
        """分词"""
        if lang == 'zh':
            words = list(text)
            words = [w for w in words if w.strip() and not w.isspace()]
        else:
            words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
        return [w for w in words if w not in self.stop_words and len(w) > 1]
    
    def _get_ngrams(self, words: List[str], n: int) -> List[Tuple]:
        """获取n-gram"""
        return list(zip(*[words[i:] for i in range(n)])) if len(words) >= n else []
    
    def analyze_document(self, text: str) -> Dict:
        """分析单个文档的信息密度"""
        lang = self._detect_language(text[:1000])
        words = self._tokenize(text, lang)
        
        if len(words) < 50:
            return {'error': '文本过短', 'word_count': len(words)}
        
        results = {
            'language': lang,
            'word_count': len(words)
        }
        
        # 词频分布
        word_freq = Counter(words)
        
        # 1. 信息熵
        total_words = len(words)
        entropy = -sum((freq / total_words) * math.log2(freq / total_words) 
                      for freq in word_freq.values())
        results['information_entropy'] = entropy
        
        # 2. 最大熵（均匀分布）
        max_entropy = math.log2(len(word_freq))
        results['max_entropy'] = max_entropy
        
        # 3. 相对熵（熵比）
        results['relative_entropy'] = entropy / max_entropy if max_entropy > 0 else 0
        
        # 4. 词汇重复率
        unique_words = set(words)
        repetition_rate = 1 - (len(unique_words) / len(words)) if words else 0
        results['repetition_rate'] = repetition_rate
        
        # 5. n-gram多样性
        for n in [2, 3, 4]:
            ngrams = self._get_ngrams(words, n)
            if ngrams:
                unique_ngrams = set(ngrams)
                results[f'{n}gram_diversity'] = len(unique_ngrams) / len(ngrams)
            else:
                results[f'{n}gram_diversity'] = 0
        
        # 6. 内容词密度（去除停用词后的词数比例）
        all_words = re.findall(r'\b[a-zA-Z]+\b', text.lower()) if lang == 'en' else list(text)
        content_word_ratio = len(words) / len(all_words) if all_words else 0
        results['content_word_ratio'] = content_word_ratio
        
        # 7. 高频词覆盖率（top 10词覆盖比例）
        top_10_words = word_freq.most_common(10)
        top_10_coverage = sum(freq for _, freq in top_10_words) / len(words) if words else 0
        results['top10_word_coverage'] = top_10_coverage
        
        # 8. Zipf定律参数估算
        freq_values = sorted(word_freq.values(), reverse=True)
        if len(freq_values) >= 10:
            log_ranks = [math.log(i + 1) for i in range(len(freq_values))]
            log_freqs = [math.log(f) for f in freq_values]
            
            # 简单线性回归估算斜率
            slope, intercept, r_value, p_value, std_err = stats.linregress(log_ranks, log_freqs)
            results['zipf_slope'] = slope
            results['zipf_r_squared'] = r_value ** 2
        else:
            results['zipf_slope'] = 0
            results['zipf_r_squared'] = 0
        
        # 9. 信息密度指数
        density_index = (
            results['relative_entropy'] * 0.3 +
            results['content_word_ratio'] * 0.3 +
            (1 - results['repetition_rate']) * 0.2 +
            results['2gram_diversity'] * 0.2
        )
        results['information_density_index'] = density_index
        
        # 10. 词汇冗余度
        results['lexical_redundancy'] = 1 - results['relative_entropy']
        
        # 11. 平均词频（每个词平均出现次数）
        results['mean_word_frequency'] = len(words) / len(unique_words) if unique_words else 0
        
        # 12. 困惑度估算（基于熵）
        results['perplexity_estimate'] = 2 ** entropy
        
        return results
    
    def analyze_batch(self, documents: List[Dict]) -> Dict:
        """批量分析文档（支持缓存）"""
        results = {}
        
        for doc in SilentProgress(documents, desc="信息密度分析", report_interval=50):
            doc_id = doc['filename']
            
            # 尝试从缓存获取
            if self.cache_manager:
                cached = self.cache_manager.get_cached_result(doc_id, 'density')
                if cached:
                    results[doc_id] = cached
                    results[doc_id]['data_source'] = doc['data_source']
                    continue
            
            # 分析文档
            result = self.analyze_document(doc['content'])
            result['data_source'] = doc['data_source']
            results[doc_id] = result
            
            # 保存到缓存
            if self.cache_manager and 'error' not in result:
                self.cache_manager.save_result(doc_id, 'density', result)
        
        return results
    
    def get_summary_statistics(self, results: Dict) -> Dict:
        """获取汇总统计"""
        metrics = ['information_entropy', 'relative_entropy', 'repetition_rate',
                   '2gram_diversity', '3gram_diversity', '4gram_diversity',
                   'content_word_ratio', 'top10_word_coverage', 'zipf_slope',
                   'information_density_index', 'lexical_redundancy',
                   'mean_word_frequency', 'perplexity_estimate']
        
        summary = {}
        for metric in metrics:
            values = [r.get(metric, 0) for r in results.values() if 'error' not in r]
            if values:
                summary[metric] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'median': np.median(values),
                    'min': np.min(values),
                    'max': np.max(values)
                }
        
        return summary


# ==================== 结构与连贯性分析器 ====================
class StructureCoherenceAnalyzer:
    """结构与连贯性分析器
    
    分析指标：
    1. 段落数量和长度分布
    2. 连接词密度
    3. 主题连贯性
    4. 文本结构标记
    5. 章节层次结构
    6. 指代链分析
    7. 段落主题一致性
    8. 结构复杂度指数
    """
    
    def __init__(self, config: LinguisticAnalysisConfig, stop_words: Set[str]):
        self.config = config
        self.stop_words = stop_words
        self.cache_manager = None  # 缓存管理器
        
        # 连接词列表
        self.connectives = {
            'addition': ['also', 'furthermore', 'moreover', 'additionally', 
                        'besides', 'in addition', 'as well as'],
            'contrast': ['however', 'nevertheless', 'nonetheless', 'yet', 
                        'but', 'on the other hand', 'in contrast', 'whereas'],
            'cause_effect': ['therefore', 'thus', 'hence', 'consequently', 
                            'as a result', 'accordingly', 'because', 'since'],
            'sequence': ['first', 'second', 'third', 'finally', 'next', 
                        'then', 'subsequently', 'meanwhile', 'afterwards'],
            'exemplification': ['for example', 'for instance', 'such as', 
                               'specifically', 'in particular', 'notably'],
            'conclusion': ['in conclusion', 'to summarize', 'overall', 
                          'in summary', 'ultimately', 'in short'],
            'condition': ['if', 'provided that', 'assuming that', 
                         'in case', 'unless', 'whether']
        }
        
        # 章节标记
        self.section_markers = [
            'chapter', 'section', 'article', 'part', 'title',
            'appendix', 'annex', 'preamble', 'introduction',
            'conclusion', 'summary', 'background', 'overview'
        ]
        
        # 结构标记词
        self.structure_markers = [
            'firstly', 'secondly', 'thirdly', 'finally', 'lastly',
            'first of all', 'in the first place', 'to begin with',
            'on the one hand', 'on the other hand', 'in addition',
            'furthermore', 'moreover', 'besides', 'also',
            'however', 'nevertheless', 'nonetheless', 'yet',
            'therefore', 'thus', 'hence', 'consequently',
            'in conclusion', 'to summarize', 'in summary'
        ]
    
    def _detect_language(self, text: str) -> str:
        """检测文本语言"""
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        total_chars = len(text.replace(' ', ''))
        if total_chars > 0 and chinese_chars / total_chars > 0.3:
            return 'zh'
        return 'en'
    
    def _split_paragraphs(self, text: str) -> List[str]:
        """分割段落"""
        paragraphs = re.split(r'\n\s*\n|\n{2,}', text)
        return [p.strip() for p in paragraphs if len(p.strip()) > 50]
    
    def _count_connectives(self, text: str) -> Dict[str, int]:
        """统计连接词"""
        text_lower = text.lower()
        counts = {}
        
        for category, connective_list in self.connectives.items():
            count = sum(1 for c in connective_list if c in text_lower)
            counts[category] = count
        
        counts['total'] = sum(counts.values())
        return counts
    
    def _count_section_markers(self, text: str) -> Dict:
        """统计章节标记"""
        text_lower = text.lower()
        
        marker_counts = {}
        for marker in self.section_markers:
            pattern = rf'\b{marker}\s*\d+'
            matches = re.findall(pattern, text_lower)
            marker_counts[marker] = len(matches)
        
        return marker_counts
    
    def _analyze_paragraph_structure(self, paragraphs: List[str]) -> Dict:
        """分析段落结构"""
        if not paragraphs:
            return {'paragraph_count': 0}
        
        lengths = [len(p.split()) for p in paragraphs]
        
        return {
            'paragraph_count': len(paragraphs),
            'mean_paragraph_length': np.mean(lengths),
            'paragraph_length_std': np.std(lengths),
            'min_paragraph_length': min(lengths),
            'max_paragraph_length': max(lengths),
            'paragraph_length_cv': np.std(lengths) / np.mean(lengths) if np.mean(lengths) > 0 else 0
        }
    
    def _analyze_coherence(self, text: str, paragraphs: List[str]) -> Dict:
        """分析文本连贯性"""
        results = {}
        
        # 1. 连接词密度
        connective_counts = self._count_connectives(text)
        words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
        word_count = len(words)
        
        results['connective_density'] = connective_counts['total'] / word_count if word_count > 0 else 0
        
        # 各类连接词密度
        for category, count in connective_counts.items():
            if category != 'total':
                results[f'{category}_connective_density'] = count / word_count if word_count > 0 else 0
        
        # 2. 结构标记密度
        structure_marker_count = sum(1 for m in self.structure_markers if m in text.lower())
        results['structure_marker_density'] = structure_marker_count / word_count if word_count > 0 else 0
        
        # 3. 章节标记
        section_counts = self._count_section_markers(text)
        results['section_marker_count'] = sum(section_counts.values())
        results['section_types'] = sum(1 for c in section_counts.values() if c > 0)
        
        # 4. 段落间主题一致性（基于关键词重叠）
        if len(paragraphs) >= 2:
            paragraph_keywords = []
            for p in paragraphs:
                words_p = [w for w in re.findall(r'\b[a-zA-Z]+\b', p.lower()) 
                          if w not in self.stop_words and len(w) > 3]
                paragraph_keywords.append(set(words_p))
            
            # 计算相邻段落关键词重叠
            overlaps = []
            for i in range(len(paragraph_keywords) - 1):
                overlap = len(paragraph_keywords[i] & paragraph_keywords[i + 1])
                union = len(paragraph_keywords[i] | paragraph_keywords[i + 1])
                jaccard = overlap / union if union > 0 else 0
                overlaps.append(jaccard)
            
            results['paragraph_coherence'] = np.mean(overlaps) if overlaps else 0
        else:
            results['paragraph_coherence'] = 0
        
        # 5. 指代词密度
        reference_words = ['this', 'that', 'these', 'those', 'it', 'they', 
                          'which', 'who', 'such', 'the above', 'the following']
        ref_count = sum(1 for w in reference_words if w in text.lower())
        results['reference_word_density'] = ref_count / word_count if word_count > 0 else 0
        
        return results
    
    def analyze_document(self, text: str) -> Dict:
        """分析单个文档的结构与连贯性"""
        lang = self._detect_language(text[:1000])
        
        # 分段
        paragraphs = self._split_paragraphs(text)
        
        if len(paragraphs) < 2:
            return {'error': '段落数量不足', 'paragraph_count': len(paragraphs)}
        
        results = {
            'language': lang
        }
        
        # 段落结构分析
        paragraph_analysis = self._analyze_paragraph_structure(paragraphs)
        results.update(paragraph_analysis)
        
        # 连贯性分析
        coherence_analysis = self._analyze_coherence(text, paragraphs)
        results.update(coherence_analysis)
        
        # 结构复杂度综合指数
        complexity_index = (
            min(results['paragraph_count'] / 20, 1) * 0.2 +
            results['connective_density'] * 50 * 0.2 +
            min(results['section_types'] / 5, 1) * 0.2 +
            results['paragraph_coherence'] * 0.2 +
            results['reference_word_density'] * 50 * 0.2
        )
        results['structure_complexity_index'] = complexity_index
        
        # 文本结构平衡性（段落长度变异系数的倒数）
        cv = results.get('paragraph_length_cv', 1)
        results['structure_balance'] = 1 / (1 + cv) if cv >= 0 else 0
        
        return results
    
    def analyze_batch(self, documents: List[Dict]) -> Dict:
        """批量分析文档（支持缓存）"""
        results = {}
        
        for doc in SilentProgress(documents, desc="结构连贯性分析", report_interval=50):
            doc_id = doc['filename']
            
            # 尝试从缓存获取
            if self.cache_manager:
                cached = self.cache_manager.get_cached_result(doc_id, 'coherence')
                if cached:
                    results[doc_id] = cached
                    results[doc_id]['data_source'] = doc['data_source']
                    continue
            
            # 分析文档
            result = self.analyze_document(doc['content'])
            result['data_source'] = doc['data_source']
            results[doc_id] = result
            
            # 保存到缓存
            if self.cache_manager and 'error' not in result:
                self.cache_manager.save_result(doc_id, 'coherence', result)
        
        return results
    
    def get_summary_statistics(self, results: Dict) -> Dict:
        """获取汇总统计"""
        metrics = ['paragraph_count', 'mean_paragraph_length', 'paragraph_length_std',
                   'paragraph_length_cv', 'connective_density', 'structure_marker_density',
                   'section_marker_count', 'section_types', 'paragraph_coherence',
                   'reference_word_density', 'structure_complexity_index', 'structure_balance']
        
        summary = {}
        for metric in metrics:
            values = [r.get(metric, 0) for r in results.values() if 'error' not in r]
            if values:
                summary[metric] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'median': np.median(values),
                    'min': np.min(values),
                    'max': np.max(values)
                }
        
        return summary


# ==================== 综合分析报告生成器 ====================
class LinguisticReportGenerator:
    """语言学分析报告生成器"""
    
    def __init__(self, config: LinguisticAnalysisConfig):
        self.config = config
    
    def generate_full_report(self, 
                            richness_results: Dict,
                            difficulty_results: Dict,
                            syntactic_results: Dict,
                            density_results: Dict,
                            coherence_results: Dict,
                            metadata_loader: MetadataLoader,
                            output_dir: Path):
        """生成完整分析报告"""
        
        # 1. 生成汇总数据
        summary_data = self._create_summary_data(
            richness_results, difficulty_results, syntactic_results,
            density_results, coherence_results
        )
        
        # 2. 保存详细结果JSON
        self._save_detailed_results(
            richness_results, difficulty_results, syntactic_results,
            density_results, coherence_results, output_dir
        )
        
        # 3. 生成可视化
        self._create_visualizations(summary_data, output_dir)
        
        # 4. 生成CSV汇总
        self._create_summary_csv(summary_data, output_dir)
        
        # 5. 生成Markdown报告
        self._create_markdown_report(summary_data, output_dir)
        
        logger.info(f"  ✓ 报告已生成: {output_dir}")
    
    def _create_summary_data(self, *all_results) -> Dict:
        """创建汇总数据"""
        documents = set()
        for results in all_results:
            documents.update(results.keys())
        
        summary = {'documents': {}}
        
        for doc_id in documents:
            doc_data = {
                'document_id': doc_id,
                'richness': {},
                'difficulty': {},
                'syntactic': {},
                'density': {},
                'coherence': {}
            }
            
            if doc_id in all_results[0]:
                doc_data['richness'] = {k: v for k, v in all_results[0][doc_id].items() 
                                       if not isinstance(v, dict)}
            if doc_id in all_results[1]:
                doc_data['difficulty'] = {k: v for k, v in all_results[1][doc_id].items() 
                                        if not isinstance(v, dict)}
            if doc_id in all_results[2]:
                doc_data['syntactic'] = {k: v for k, v in all_results[2][doc_id].items() 
                                        if not isinstance(v, dict)}
            if doc_id in all_results[3]:
                doc_data['density'] = {k: v for k, v in all_results[3][doc_id].items() 
                                      if not isinstance(v, dict)}
            if doc_id in all_results[4]:
                doc_data['coherence'] = {k: v for k, v in all_results[4][doc_id].items() 
                                        if not isinstance(v, dict)}
            
            summary['documents'][doc_id] = doc_data
        
        return summary
    
    def _save_detailed_results(self, *all_results_and_dir):
        """保存详细结果"""
        output_dir = all_results_and_dir[-1]
        results = all_results_and_dir[:-1]
        
        names = ['vocabulary_richness', 'vocabulary_difficulty', 
                'syntactic_complexity', 'information_density', 'structure_coherence']
        
        for name, result in zip(names, results):
            # 转换numpy类型
            result_serializable = self._convert_to_serializable(result)
            
            with open(output_dir / f'{name}_results.json', 'w', encoding='utf-8') as f:
                json.dump(result_serializable, f, ensure_ascii=False, indent=2)
    
    def _convert_to_serializable(self, obj):
        """转换为可序列化格式"""
        if isinstance(obj, dict):
            return {k: self._convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_serializable(v) for v in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        else:
            return obj
    
    def _create_visualizations(self, summary_data: Dict, output_dir: Path):
        """创建可视化图表"""
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        
        vis_dir = output_dir / 'visualizations'
        vis_dir.mkdir(exist_ok=True)
        
        # 1. 综合指标雷达图
        self._create_radar_chart(summary_data, vis_dir)
        
        # 2. 各维度分布箱线图
        self._create_boxplots(summary_data, vis_dir)
        
        # 3. 指标相关性热力图
        self._create_correlation_heatmap(summary_data, vis_dir)
    
    def _create_radar_chart(self, summary_data: Dict, output_dir: Path):
        """创建雷达图"""
        if not PLOTLY_AVAILABLE:
            logger.warning("⚠️ Plotly未安装，跳过雷达图生成")
            return
        
        # 计算各维度的归一化均值
        metrics = {
            '词汇丰富度': ['ttr', 'herdan_c', 'lexical_density'],
            '词汇难度': ['vocabulary_complexity_index', 'academic_word_ratio', 'policy_term_density'],
            '句法复杂度': ['syntactic_complexity_index', 'mean_sentence_length_words', 'complex_sentence_ratio'],
            '信息密度': ['information_density_index', 'relative_entropy', 'content_word_ratio'],
            '结构连贯性': ['structure_complexity_index', 'paragraph_coherence', 'connective_density']
        }
        
        # 随机抽取一些文档用于展示
        doc_ids = list(summary_data['documents'].keys())[:50]
        
        if not doc_ids:
            return
        
        # 计算各维度平均分
        dimension_scores = {}
        for dim, metric_list in metrics.items():
            scores = []
            for doc_id in doc_ids:
                doc_data = summary_data['documents'][doc_id]
                dim_key = {
                    '词汇丰富度': 'richness',
                    '词汇难度': 'difficulty', 
                    '句法复杂度': 'syntactic',
                    '信息密度': 'density',
                    '结构连贯性': 'coherence'
                }[dim]
                
                values = []
                for m in metric_list:
                    val = doc_data.get(dim_key, {}).get(m, 0)
                    if isinstance(val, (int, float)):
                        values.append(val)
                
                if values:
                    scores.append(np.mean(values))
            
            if scores:
                dimension_scores[dim] = np.mean(scores)
        
        # 归一化到0-1范围
        max_score = max(dimension_scores.values()) if dimension_scores else 1
        normalized_scores = {k: v / max_score for k, v in dimension_scores.items()}
        
        # 创建雷达图
        categories = list(normalized_scores.keys())
        values = list(normalized_scores.values())
        values += values[:1]  # 闭合
        
        fig = go.Figure(data=go.Scatterpolar(
            r=values,
            theta=categories + [categories[0]],
            fill='toself',
            name='平均得分',
            line_color='#2196F3'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            showlegend=True,
            title='文本语言学特征综合评分<br><sup>(Linguistic Feature Comprehensive Score)</sup>',
            paper_bgcolor='#F5FAFF',
            font=dict(color='#1565C0')
        )
        
        fig.write_html(str(output_dir / 'comprehensive_radar_chart.html'))
    
    def _create_boxplots(self, summary_data: Dict, output_dir: Path):
        """创建箱线图"""
        if not PLOTLY_AVAILABLE:
            logger.warning("⚠️ Plotly未安装，跳过箱线图生成")
            return
        
        # 选择关键指标
        key_metrics = {
            '词汇丰富度': ('richness', 'ttr'),
            '词汇难度': ('difficulty', 'vocabulary_complexity_index'),
            '句法复杂度': ('syntactic', 'syntactic_complexity_index'),
            '信息密度': ('density', 'information_density_index'),
            '结构连贯性': ('coherence', 'structure_complexity_index')
        }
        
        fig = make_subplots(
            rows=1, cols=5,
            subplot_titles=list(key_metrics.keys())
        )
        
        for i, (name, (dim_key, metric)) in enumerate(key_metrics.items()):
            values = []
            for doc_data in summary_data['documents'].values():
                val = doc_data.get(dim_key, {}).get(metric, None)
                if val is not None and isinstance(val, (int, float)):
                    values.append(val)
            
            if values:
                fig.add_trace(
                    go.Box(y=values, name=name, marker_color='#2196F3'),
                    row=1, col=i+1
                )
        
        fig.update_layout(
            title_text='各维度关键指标分布<br><sup>(Distribution of Key Metrics)</sup>',
            showlegend=False,
            height=500,
            paper_bgcolor='#F5FAFF',
            font=dict(color='#1565C0')
        )
        
        fig.write_html(str(output_dir / 'metrics_boxplot.html'))
    
    def _create_correlation_heatmap(self, summary_data: Dict, output_dir: Path):
        """创建相关性热力图"""
        if not PLOTLY_AVAILABLE:
            logger.warning("⚠️ Plotly未安装，跳过热力图生成")
            return
        
        # 收集所有数值指标
        all_metrics = []
        metric_names = []
        
        # 从每个文档收集指标
        doc_metrics_list = []
        for doc_data in summary_data['documents'].values():
            doc_metrics = {}
            for dim in ['richness', 'difficulty', 'syntactic', 'density', 'coherence']:
                for k, v in doc_data.get(dim, {}).items():
                    if isinstance(v, (int, float)) and not k.endswith('_distribution'):
                        key = f"{dim[:3]}_{k}"[:20]
                        doc_metrics[key] = v
            doc_metrics_list.append(doc_metrics)
        
        # 选择常见指标
        if doc_metrics_list:
            common_keys = set(doc_metrics_list[0].keys())
            for dm in doc_metrics_list[1:]:
                common_keys &= set(dm.keys())
            
            common_keys = sorted(list(common_keys))[:15]  # 限制数量
            
            if len(common_keys) >= 5:
                # 创建数据矩阵
                data_matrix = []
                for key in common_keys:
                    row = [dm.get(key, 0) for dm in doc_metrics_list]
                    data_matrix.append(row)
                
                data_matrix = np.array(data_matrix)
                
                # 计算相关性矩阵
                if data_matrix.shape[1] > 1:
                    corr_matrix = np.corrcoef(data_matrix)
                    
                    fig = ff.create_annotated_heatmap(
                        z=corr_matrix,
                        x=common_keys,
                        y=common_keys,
                        colorscale='RdBu',
                        showscale=True,
                        zmin=-1, zmax=1
                    )
                    
                    fig.update_layout(
                        title='关键指标相关性矩阵<br><sup>(Correlation Matrix of Key Metrics)</sup>',
                        paper_bgcolor='#F5FAFF',
                        font=dict(color='#1565C0'),
                        width=800, height=800
                    )
                    
                    fig.write_html(str(output_dir / 'correlation_heatmap.html'))
    
    def _create_summary_csv(self, summary_data: Dict, output_dir: Path):
        """创建汇总CSV"""
        rows = []
        
        for doc_id, doc_data in summary_data['documents'].items():
            row = {'document_id': doc_id}
            
            for dim in ['richness', 'difficulty', 'syntactic', 'density', 'coherence']:
                dim_data = doc_data.get(dim, {})
                for k, v in dim_data.items():
                    if isinstance(v, (int, float)):
                        row[f"{dim}_{k}"] = v
            
            rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_csv(output_dir / 'linguistic_features_summary.csv', index=False, encoding='utf-8')
    
    def _create_markdown_report(self, summary_data: Dict, output_dir: Path):
        """创建Markdown报告"""
        total_docs = len(summary_data['documents'])
        
        report = f"""# 文本语言学特征分析报告

## 分析概要

- **分析文档数量**: {total_docs}
- **分析日期**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **筛选条件**: {self.config.get_filter_summary()}

## 分析维度

### 1. 词汇丰富度 (Vocabulary Richness)

分析文本的词汇多样性，包括：
- **TTR (Type-Token Ratio)**: 词汇类型与总词数的比值
- **RTTR (Root TTR)**: 修正后的TTR，减少文本长度影响
- **Herdan's C**: 对数形式的词汇丰富度指标
- **Maas's Index**: 词汇丰富度的另一种测量方法
- **Honoré's Statistic**: 考虑hapax legomena的指标
- **Yule's K**: 衡量词汇重复度

### 2. 词汇难度 (Vocabulary Difficulty)

评估文本的词汇复杂度，针对治理文本特点：
- **平均词长**: 词汇长度的平均水平
- **长词比例**: 超过6个字母的词汇比例
- **学术词汇比例**: 学术性词汇的占比
- **政策术语密度**: 政策相关术语的密度
- **法律术语密度**: 法律专业术语的密度
- **词汇复杂度指数**: 综合词汇难度指标

### 3. 句法复杂度 (Syntactic Complexity)

分析句子的结构和复杂程度：
- **平均句长**: 句子的平均词数
- **句子长度变异性**: 句子长度的标准差
- **复杂句比例**: 包含从句等复杂结构的句子比例
- **句法复杂度指数**: 综合句法复杂度指标
- **依存句法深度**: 句法树的平均深度

### 4. 信息密度 (Information Density)

评估文本的信息含量和可预测性：
- **信息熵**: 文本的信息量测量
- **相对熵**: 实际熵与最大熵的比值
- **词汇重复率**: 词汇重复出现的比例
- **n-gram多样性**: 连续词组的多样性
- **信息密度指数**: 综合信息密度指标

### 5. 结构与连贯性 (Structure & Coherence)

分析文本的组织结构和连贯性：
- **段落数量和分布**: 段落的数量和长度分布
- **连接词密度**: 各类连接词的使用频率
- **章节结构**: 章节标记的使用情况
- **段落连贯性**: 相邻段落主题的一致性
- **结构复杂度指数**: 综合结构指标

## 输出文件

- `linguistic_features_summary.csv`: 所有文档的语言学特征汇总
- `vocabulary_richness_results.json`: 词汇丰富度详细结果
- `vocabulary_difficulty_results.json`: 词汇难度详细结果
- `syntactic_complexity_results.json`: 句法复杂度详细结果
- `information_density_results.json`: 信息密度详细结果
- `structure_coherence_results.json`: 结构连贯性详细结果
- `visualizations/`: 可视化图表目录

## 使用说明

本分析系统复用了叙事分析系统的配置框架和数据源，支持相同的筛选功能：
- 按国家/地区筛选
- 按组织类型筛选
- 按文档类型筛选
- 按数据源筛选

分析方法参考了计算语言学和文本分析的学术文献，特别针对政策治理文本的特点进行了优化。
"""
        
        with open(output_dir / 'linguistic_analysis_report.md', 'w', encoding='utf-8') as f:
            f.write(report)


# ==================== 缓存管理器 ====================
class LinguisticCacheManager:
    """语言学分析缓存管理器"""
    
    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        self.cache_file = cache_dir / "linguistic_analysis_cache.pkl"
        self.cache_stats = {'hits': 0, 'misses': 0}
    
    def _get_doc_hash(self, doc_id: str, analysis_type: str) -> str:
        """生成文档分析的缓存键"""
        return f"{analysis_type}_{doc_id}"
    
    def get_cached_result(self, doc_id: str, analysis_type: str) -> Optional[Dict]:
        """获取缓存的分析结果"""
        try:
            if self.cache_file.exists():
                with open(self.cache_file, 'rb') as f:
                    cache_data = pickle.load(f)
                key = self._get_doc_hash(doc_id, analysis_type)
                if key in cache_data:
                    self.cache_stats['hits'] += 1
                    return cache_data[key]
        except Exception as e:
            logger.debug(f"缓存读取失败: {e}")
        self.cache_stats['misses'] += 1
        return None
    
    def save_result(self, doc_id: str, analysis_type: str, result: Dict):
        """保存分析结果到缓存"""
        try:
            cache_data = {}
            if self.cache_file.exists():
                with open(self.cache_file, 'rb') as f:
                    cache_data = pickle.load(f)
            
            key = self._get_doc_hash(doc_id, analysis_type)
            cache_data[key] = result
            
            with open(self.cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
        except Exception as e:
            logger.debug(f"缓存保存失败: {e}")
    
    def get_stats(self) -> Dict:
        """获取缓存统计"""
        total = self.cache_stats['hits'] + self.cache_stats['misses']
        hit_rate = self.cache_stats['hits'] / total if total > 0 else 0
        return {
            'hit_rate': hit_rate,
            'hits': self.cache_stats['hits'],
            'misses': self.cache_stats['misses']
        }


# ==================== 主分析流程 ====================
def run_full_analysis(config: LinguisticAnalysisConfig = None):
    """运行完整的语言学特征分析
    
    Args:
        config: 可选的配置对象，如果不提供则创建默认配置
    """
    global progress
    
    # 重置进度追踪器
    progress = ProgressTracker()
    
    logger.info("═" * 80)
    logger.info("📊 文本语言学特征分析系统启动")
    logger.info(f"📅 开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("═" * 80)
    
    # ========== 阶段1: 初始化 ==========
    progress.start_stage(0)
    if config is None:
        config = LinguisticAnalysisConfig()
    
    # 初始化缓存管理器
    cache_manager = LinguisticCacheManager(config.cache_dir)
    progress.end_stage()
    
    # ========== 阶段2: 加载停用词 ==========
    progress.start_stage(1)
    stopwords_loader = StopwordsLoader(config.stopwords_paths)
    stop_words = stopwords_loader.load_all_stopwords()
    progress.end_stage()
    
    # ========== 阶段3: 加载元数据 ==========
    progress.start_stage(2)
    metadata_loader = MetadataLoader(config.metadata_dir)
    progress.end_stage()
    
    # ========== 阶段4: 加载文档 ==========
    progress.start_stage(3)
    doc_loader = DocumentLoader(
        config.agora_fulltext_dir,
        config.original_data_dir,
        metadata_loader,
        config
    )
    documents = doc_loader.load_documents()
    
    if not documents:
        logger.error("❌ 没有找到有效文档")
        return
    progress.end_stage()
    
    # ========== 阶段5: 词汇丰富度分析 ==========
    progress.start_stage(4)
    richness_analyzer = VocabularyRichnessAnalyzer(config, stop_words)
    richness_analyzer.cache_manager = cache_manager  # 设置缓存管理器
    richness_results = richness_analyzer.analyze_batch(documents)
    richness_summary = richness_analyzer.get_summary_statistics(richness_results)
    progress.log_metric("平均TTR", f"{richness_summary.get('ttr', {}).get('mean', 0):.4f}")
    progress.end_stage()
    
    # ========== 阶段6: 词汇难度分析 ==========
    progress.start_stage(5)
    difficulty_analyzer = VocabularyDifficultyAnalyzer(config, stop_words)
    difficulty_analyzer.cache_manager = cache_manager
    difficulty_results = difficulty_analyzer.analyze_batch(documents)
    difficulty_summary = difficulty_analyzer.get_summary_statistics(difficulty_results)
    progress.log_metric("平均词汇复杂度指数", 
                       f"{difficulty_summary.get('vocabulary_complexity_index', {}).get('mean', 0):.4f}")
    progress.end_stage()
    
    # ========== 阶段7: 句法复杂度分析 ==========
    progress.start_stage(6)
    syntactic_analyzer = SyntacticComplexityAnalyzer(config, stop_words)
    syntactic_analyzer.cache_manager = cache_manager
    syntactic_results = syntactic_analyzer.analyze_batch(documents)
    syntactic_summary = syntactic_analyzer.get_summary_statistics(syntactic_results)
    progress.log_metric("平均句法复杂度指数", 
                       f"{syntactic_summary.get('syntactic_complexity_index', {}).get('mean', 0):.4f}")
    progress.end_stage()
    
    # ========== 阶段8: 信息密度分析 ==========
    progress.start_stage(7)
    density_analyzer = InformationDensityAnalyzer(config, stop_words)
    density_analyzer.cache_manager = cache_manager
    density_results = density_analyzer.analyze_batch(documents)
    density_summary = density_analyzer.get_summary_statistics(density_results)
    progress.log_metric("平均信息密度指数", 
                       f"{density_summary.get('information_density_index', {}).get('mean', 0):.4f}")
    progress.end_stage()
    
    # ========== 阶段9: 结构连贯性分析 ==========
    progress.start_stage(8)
    coherence_analyzer = StructureCoherenceAnalyzer(config, stop_words)
    coherence_analyzer.cache_manager = cache_manager
    coherence_results = coherence_analyzer.analyze_batch(documents)
    coherence_summary = coherence_analyzer.get_summary_statistics(coherence_results)
    progress.log_metric("平均结构复杂度指数", 
                       f"{coherence_summary.get('structure_complexity_index', {}).get('mean', 0):.4f}")
    progress.end_stage()
    
    # ========== 阶段10: 保存结果 ==========
    progress.start_stage(9)
    report_generator = LinguisticReportGenerator(config)
    report_generator.generate_full_report(
        richness_results, difficulty_results, syntactic_results,
        density_results, coherence_results, metadata_loader,
        config.output_dir
    )
    
    # 输出缓存统计
    cache_stats = cache_manager.get_stats()
    logger.info(f"💾 缓存统计: 命中率 {cache_stats['hit_rate']:.1%}, "
               f"命中 {cache_stats['hits']}, 未命中 {cache_stats['misses']}")
    progress.end_stage()
    
    # ========== 阶段11: 生成报告 ==========
    progress.start_stage(10)
    logger.info(f"\n📁 输出目录: {config.output_dir}")
    progress.end_stage()
    
    # 完成
    progress.finish()
    
    return {
        'config': config,
        'richness_results': richness_results,
        'difficulty_results': difficulty_results,
        'syntactic_results': syntactic_results,
        'density_results': density_results,
        'coherence_results': coherence_results
    }


def run_analysis_with_country_filter(countries: List[str]):
    """运行带国家筛选的分析
    
    Args:
        countries: 国家/地区列表，如 ['美国', '中国']
    
    Returns:
        分析结果字典
    """
    logger.info(f"🌍 设置国家筛选: {countries}")
    
    # 创建配置并设置筛选条件（必须在初始化前设置）
    config = LinguisticAnalysisConfig.__new__(LinguisticAnalysisConfig)
    config._init_paths()
    config._init_filters()
    config._init_linguistic_settings()
    
    # 设置筛选条件
    config.filter_countries = countries
    
    # 创建输出目录（应用筛选条件后）
    config._setup_output_dirs()
    
    # 显示配置信息
    logger.info(f"📁 输出目录: {config.output_dir}")
    logger.info(f"🌍 国家筛选: {config.filter_countries}")
    
    return run_full_analysis(config)


def run_analysis_with_org_type_filter(org_types: List[str]):
    """运行带组织类型筛选的分析
    
    Args:
        org_types: 组织类型列表，如 ['government', 'tech_company']
    
    Returns:
        分析结果字典
    """
    logger.info(f"🏢 设置组织类型筛选: {org_types}")
    
    # 创建配置并设置筛选条件
    config = LinguisticAnalysisConfig.__new__(LinguisticAnalysisConfig)
    config._init_paths()
    config._init_filters()
    config._init_linguistic_settings()
    
    # 设置筛选条件
    config.filter_org_types = org_types
    
    # 创建输出目录
    config._setup_output_dirs()
    
    # 显示配置信息
    logger.info(f"📁 输出目录: {config.output_dir}")
    logger.info(f"🏢 组织类型筛选: {config.filter_org_types}")
    
    return run_full_analysis(config)


def run_analysis_with_filters(countries: List[str] = None, 
                              org_types: List[str] = None,
                              doc_types: List[str] = None,
                              data_sources: List[str] = None):
    """运行带多条件筛选的分析
    
    Args:
        countries: 国家/地区列表
        org_types: 组织类型列表
        doc_types: 文档类型列表
        data_sources: 数据源列表
    
    Returns:
        分析结果字典
    """
    # 创建配置并设置筛选条件
    config = LinguisticAnalysisConfig.__new__(LinguisticAnalysisConfig)
    config._init_paths()
    config._init_filters()
    config._init_linguistic_settings()
    
    # 设置筛选条件
    if countries:
        config.filter_countries = countries
        logger.info(f"🌍 国家筛选: {countries}")
    if org_types:
        config.filter_org_types = org_types
        logger.info(f"🏢 组织类型筛选: {org_types}")
    if doc_types:
        config.filter_doc_types = doc_types
        logger.info(f"📄 文档类型筛选: {doc_types}")
    if data_sources:
        config.filter_data_sources = data_sources
        logger.info(f"📂 数据源筛选: {data_sources}")
    
    # 创建输出目录
    config._setup_output_dirs()
    
    # 显示配置信息
    logger.info(f"📁 输出目录: {config.output_dir}")
    
    return run_full_analysis(config)


def list_available_filters():
    """列出可用的筛选选项"""
    config = LinguisticAnalysisConfig()
    metadata_loader = MetadataLoader(config.metadata_dir)
    
    logger.info("\n📋 可用的筛选选项:")
    logger.info("─" * 40)
    
    countries = metadata_loader.metadata_cache
    country_set = set()
    org_type_set = set()
    doc_type_set = set()
    
    for meta in countries.values():
        if meta.get('level1_country_or_org'):
            country_set.add(meta['level1_country_or_org'])
        if meta.get('level2_org_type'):
            org_type_set.add(meta['level2_org_type'])
        if meta.get('level3_doc_type'):
            doc_type_set.add(meta['level3_doc_type'])
    
    logger.info(f"🌍 国家/地区: {sorted(country_set)}")
    logger.info(f"🏢 组织类型: {sorted(org_type_set)}")
    logger.info(f"📄 文档类型: {sorted(doc_type_set)}")


# ==================== 时间序列分析器 ====================
class TimeSeriesAnalyzer:
    """时间序列分析器 - 按年份分析语言学特征演变"""
    
    def __init__(self, config: LinguisticAnalysisConfig, metadata_loader: MetadataLoader):
        self.config = config
        self.metadata_loader = metadata_loader
    
    def group_documents_by_year(self, documents: List[Dict]) -> Dict[int, List[Dict]]:
        """按年份分组文档"""
        year_groups = {}
        
        for doc in documents:
            year = self.metadata_loader.get_year(doc['filename'], doc['data_source'])
            if year and 1900 < year < 2100:
                if year not in year_groups:
                    year_groups[year] = []
                year_groups[year].append(doc)
        
        return dict(sorted(year_groups.items()))
    
    def analyze_year(self, year: int, docs: List[Dict], 
                    richness_results: Dict, difficulty_results: Dict,
                    syntactic_results: Dict, density_results: Dict,
                    coherence_results: Dict) -> Dict:
        """分析单一年份的特征"""
        doc_ids = [d['filename'] for d in docs]
        
        def aggregate_results(results: Dict, doc_ids: List[str]) -> Dict:
            """聚合结果"""
            metrics = {}
            valid_count = 0
            
            for doc_id in doc_ids:
                if doc_id in results and 'error' not in results[doc_id]:
                    valid_count += 1
                    for k, v in results[doc_id].items():
                        if isinstance(v, (int, float)) and k not in ['data_source']:
                            if k not in metrics:
                                metrics[k] = []
                            metrics[k].append(v)
            
            # 计算均值
            aggregated = {'doc_count': valid_count}
            for k, values in metrics.items():
                if values:
                    aggregated[f"{k}_mean"] = np.mean(values)
                    aggregated[f"{k}_std"] = np.std(values)
                    aggregated[f"{k}_median"] = np.median(values)
            
            return aggregated
        
        return {
            'year': year,
            'doc_count': len(docs),
            'richness': aggregate_results(richness_results, doc_ids),
            'difficulty': aggregate_results(difficulty_results, doc_ids),
            'syntactic': aggregate_results(syntactic_results, doc_ids),
            'density': aggregate_results(density_results, doc_ids),
            'coherence': aggregate_results(coherence_results, doc_ids)
        }
    
    def analyze_time_series(self, documents: List[Dict],
                           richness_results: Dict, difficulty_results: Dict,
                           syntactic_results: Dict, density_results: Dict,
                           coherence_results: Dict) -> Dict:
        """分析时间序列"""
        year_groups = self.group_documents_by_year(documents)
        
        results = {
            'yearly_data': {},
            'trend_analysis': {},
            'statistics': {}
        }
        
        # 分析每年数据
        for year, docs in SilentProgress(year_groups.items(), desc="时间序列分析"):
            year_result = self.analyze_year(
                year, docs, richness_results, difficulty_results,
                syntactic_results, density_results, coherence_results
            )
            results['yearly_data'][year] = year_result
        
        # 趋势分析
        results['trend_analysis'] = self._analyze_trends(results['yearly_data'])
        
        # 统计摘要
        results['statistics'] = {
            'year_range': list(year_groups.keys()),
            'total_years': len(year_groups),
            'docs_per_year': {y: len(d) for y, d in year_groups.items()}
        }
        
        return results
    
    def _analyze_trends(self, yearly_data: Dict) -> Dict:
        """分析指标趋势"""
        trends = {}
        
        years = sorted(yearly_data.keys())
        if len(years) < 2:
            return trends
        
        # 关键指标趋势
        key_metrics = {
            'richness': ['ttr_mean', 'rttr_mean', 'herdan_c_mean'],
            'difficulty': ['vocabulary_complexity_index_mean', 'avg_word_length_mean'],
            'syntactic': ['syntactic_complexity_index_mean', 'avg_sentence_length_mean'],
            'density': ['information_density_index_mean', 'entropy_mean'],
            'coherence': ['structure_complexity_index_mean', 'connective_density_mean']
        }
        
        for dim, metrics in key_metrics.items():
            for metric in metrics:
                values = []
                for year in years:
                    val = yearly_data[year].get(dim, {}).get(metric)
                    if val is not None:
                        values.append((year, val))
                
                if len(values) >= 2:
                    years_arr = np.array([v[0] for v in values])
                    vals_arr = np.array([v[1] for v in values])
                    
                    # 线性回归计算趋势
                    slope, intercept = np.polyfit(years_arr, vals_arr, 1)
                    correlation = np.corrcoef(years_arr, vals_arr)[0, 1] if len(vals_arr) > 1 else 0
                    
                    trend_key = f"{dim}_{metric.replace('_mean', '')}"
                    trends[trend_key] = {
                        'slope': slope,
                        'direction': 'increasing' if slope > 0 else 'decreasing',
                        'correlation': correlation,
                        'start_value': vals_arr[0],
                        'end_value': vals_arr[-1],
                        'change_pct': (vals_arr[-1] - vals_arr[0]) / vals_arr[0] * 100 if vals_arr[0] != 0 else 0
                    }
        
        return trends


# ==================== 国别比较分析器 ====================
class CountryComparisonAnalyzer:
    """国别比较分析器 - 比较不同国家/地区的语言学特征"""
    
    def __init__(self, config: LinguisticAnalysisConfig, metadata_loader: MetadataLoader):
        self.config = config
        self.metadata_loader = metadata_loader
    
    def group_documents_by_country(self, documents: List[Dict]) -> Dict[str, List[Dict]]:
        """按国家/地区分组文档"""
        country_groups = {}
        
        for doc in documents:
            country = self.metadata_loader.get_country(doc['filename'], doc['data_source'])
            if country and country != 'unknown':
                if country not in country_groups:
                    country_groups[country] = []
                country_groups[country].append(doc)
        
        # 按文档数量排序
        return dict(sorted(country_groups.items(), key=lambda x: -len(x[1])))
    
    def analyze_country(self, country: str, docs: List[Dict],
                       richness_results: Dict, difficulty_results: Dict,
                       syntactic_results: Dict, density_results: Dict,
                       coherence_results: Dict) -> Dict:
        """分析单一国家的特征"""
        doc_ids = [d['filename'] for d in docs]
        
        def aggregate_results(results: Dict, doc_ids: List[str]) -> Dict:
            """聚合结果"""
            metrics = {}
            valid_count = 0
            
            for doc_id in doc_ids:
                if doc_id in results and 'error' not in results[doc_id]:
                    valid_count += 1
                    for k, v in results[doc_id].items():
                        if isinstance(v, (int, float)) and k not in ['data_source']:
                            if k not in metrics:
                                metrics[k] = []
                            metrics[k].append(v)
            
            aggregated = {'doc_count': valid_count}
            for k, values in metrics.items():
                if values:
                    aggregated[f"{k}_mean"] = np.mean(values)
                    aggregated[f"{k}_std"] = np.std(values)
                    aggregated[f"{k}_median"] = np.median(values)
                    aggregated[f"{k}_min"] = np.min(values)
                    aggregated[f"{k}_max"] = np.max(values)
            
            return aggregated
        
        # 提取年份分布
        years = [self.metadata_loader.get_year(d['filename'], d['data_source']) for d in docs]
        years = [y for y in years if y and 1900 < y < 2100]
        
        return {
            'country': country,
            'doc_count': len(docs),
            'year_range': [min(years), max(years)] if years else None,
            'richness': aggregate_results(richness_results, doc_ids),
            'difficulty': aggregate_results(difficulty_results, doc_ids),
            'syntactic': aggregate_results(syntactic_results, doc_ids),
            'density': aggregate_results(density_results, doc_ids),
            'coherence': aggregate_results(coherence_results, doc_ids)
        }
    
    def analyze_countries(self, documents: List[Dict],
                         richness_results: Dict, difficulty_results: Dict,
                         syntactic_results: Dict, density_results: Dict,
                         coherence_results: Dict,
                         min_docs: int = 5) -> Dict:
        """分析所有国家"""
        country_groups = self.group_documents_by_country(documents)
        
        results = {
            'country_data': {},
            'comparison_matrix': {},
            'rankings': {},
            'statistics': {}
        }
        
        # 分析每个国家（过滤文档数量太少的国家）
        for country, docs in SilentProgress(country_groups.items(), desc="国别比较分析"):
            if len(docs) >= min_docs:
                country_result = self.analyze_country(
                    country, docs, richness_results, difficulty_results,
                    syntactic_results, density_results, coherence_results
                )
                results['country_data'][country] = country_result
        
        # 生成比较矩阵
        results['comparison_matrix'] = self._create_comparison_matrix(results['country_data'])
        
        # 生成排名
        results['rankings'] = self._create_rankings(results['country_data'])
        
        # 统计摘要
        results['statistics'] = {
            'total_countries': len(results['country_data']),
            'docs_per_country': {c: d['doc_count'] for c, d in results['country_data'].items()},
            'filtered_countries': len(country_groups) - len(results['country_data'])
        }
        
        return results
    
    def _create_comparison_matrix(self, country_data: Dict) -> Dict:
        """创建比较矩阵"""
        matrix = {}
        
        # 关键指标
        key_metrics = [
            ('richness', 'ttr_mean'),
            ('richness', 'herdan_c_mean'),
            ('difficulty', 'vocabulary_complexity_index_mean'),
            ('syntactic', 'syntactic_complexity_index_mean'),
            ('density', 'information_density_index_mean'),
            ('coherence', 'structure_complexity_index_mean')
        ]
        
        for dim, metric in key_metrics:
            matrix[f"{dim}_{metric}"] = {}
            for country, data in country_data.items():
                value = data.get(dim, {}).get(metric)
                if value is not None:
                    matrix[f"{dim}_{metric}"][country] = value
        
        return matrix
    
    def _create_rankings(self, country_data: Dict) -> Dict:
        """创建各指标排名"""
        rankings = {}
        
        # 词汇丰富度排名（TTR越高越好）
        ttr_data = [(c, d.get('richness', {}).get('ttr_mean', 0)) 
                   for c, d in country_data.items()]
        rankings['vocabulary_richness'] = sorted(ttr_data, key=lambda x: -x[1])
        
        # 词汇复杂度排名
        vocab_data = [(c, d.get('difficulty', {}).get('vocabulary_complexity_index_mean', 0))
                     for c, d in country_data.items()]
        rankings['vocabulary_complexity'] = sorted(vocab_data, key=lambda x: -x[1])
        
        # 句法复杂度排名
        syntactic_data = [(c, d.get('syntactic', {}).get('syntactic_complexity_index_mean', 0))
                         for c, d in country_data.items()]
        rankings['syntactic_complexity'] = sorted(syntactic_data, key=lambda x: -x[1])
        
        # 信息密度排名
        density_data = [(c, d.get('density', {}).get('information_density_index_mean', 0))
                       for c, d in country_data.items()]
        rankings['information_density'] = sorted(density_data, key=lambda x: -x[1])
        
        # 结构复杂度排名
        structure_data = [(c, d.get('coherence', {}).get('structure_complexity_index_mean', 0))
                         for c, d in country_data.items()]
        rankings['structure_complexity'] = sorted(structure_data, key=lambda x: -x[1])
        
        return rankings


# ==================== 比较分析可视化器 ====================
class ComparisonVisualizer:
    """比较分析可视化器"""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir / "comparisons"
        self.output_dir.mkdir(exist_ok=True, parents=True)
    
    def create_time_series_plots(self, time_series_results: Dict):
        """创建时间序列图表"""
        if not PLOTLY_AVAILABLE:
            logger.warning("⚠️ Plotly未安装，跳过可视化生成")
            return
        
        years = sorted(time_series_results['yearly_data'].keys())
        
        if len(years) < 2:
            logger.warning("年份不足，无法生成时间序列图表")
            return
        
        # 关键指标时间序列
        metrics_config = [
            ('richness', 'ttr_mean', '词汇丰富度 (TTR)', 'Vocabulary Richness (TTR)'),
            ('richness', 'herdan_c_mean', "Herdan's C", "Herdan's C Index"),
            ('difficulty', 'vocabulary_complexity_index_mean', '词汇复杂度指数', 'Vocabulary Complexity Index'),
            ('syntactic', 'syntactic_complexity_index_mean', '句法复杂度指数', 'Syntactic Complexity Index'),
            ('density', 'information_density_index_mean', '信息密度指数', 'Information Density Index'),
            ('coherence', 'structure_complexity_index_mean', '结构复杂度指数', 'Structure Complexity Index')
        ]
        
        # 创建多子图
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=[f"{cn} / {en}" for cn, en in [(m[2], m[3]) for m in metrics_config]],
            vertical_spacing=0.12,
            horizontal_spacing=0.1
        )
        
        colors = ['#1565C0', '#D32F2F', '#388E3C', '#F57C00', '#7B1FA2', '#00796B']
        
        for idx, (dim, metric, cn_name, en_name) in enumerate(metrics_config):
            row = idx // 2 + 1
            col = idx % 2 + 1
            
            values = []
            valid_years = []
            for year in years:
                val = time_series_results['yearly_data'][year].get(dim, {}).get(metric)
                if val is not None:
                    values.append(val)
                    valid_years.append(year)
            
            if values:
                fig.add_trace(
                    go.Scatter(
                        x=valid_years, y=values,
                        mode='lines+markers',
                        name=cn_name,
                        line=dict(color=colors[idx], width=2),
                        marker=dict(size=8),
                        showlegend=False
                    ),
                    row=row, col=col
                )
                
                # 添加趋势线
                if len(values) >= 3:
                    z = np.polyfit(valid_years, values, 1)
                    p = np.poly1d(z)
                    fig.add_trace(
                        go.Scatter(
                            x=valid_years, y=p(valid_years),
                            mode='lines',
                            name='趋势',
                            line=dict(color=colors[idx], width=1, dash='dash'),
                            showlegend=False
                        ),
                        row=row, col=col
                    )
        
        fig.update_layout(
            title='语言学特征时间演变<br><sup>(Temporal Evolution of Linguistic Features)</sup>',
            height=900,
            paper_bgcolor='#F5FAFF',
            font=dict(color='#1565C0')
        )
        
        fig.update_xaxes(title_text='年份 Year')
        fig.update_yaxes(title_text='指标值 Value')
        
        fig.write_html(str(self.output_dir / 'time_series_evolution.html'))
        
        # 文档数量时间分布
        doc_counts = [time_series_results['yearly_data'][y]['doc_count'] for y in years]
        
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            x=years, y=doc_counts,
            marker_color='#1565C0',
            text=doc_counts,
            textposition='outside'
        ))
        
        fig2.update_layout(
            title='文档数量年度分布<br><sup>(Document Distribution by Year)</sup>',
            xaxis_title='年份 Year',
            yaxis_title='文档数量 Document Count',
            paper_bgcolor='#F5FAFF',
            font=dict(color='#1565C0'),
            height=400
        )
        
        fig2.write_html(str(self.output_dir / 'document_distribution_by_year.html'))
    
    def create_country_comparison_plots(self, country_results: Dict, top_n: int = 15):
        """创建国别比较图表"""
        if not PLOTLY_AVAILABLE:
            logger.warning("⚠️ Plotly未安装，跳过可视化生成")
            return
        
        country_data = country_results['country_data']
        
        if not country_data:
            logger.warning("没有国家数据，无法生成比较图表")
            return
        
        # 雷达图 - Top N 国家
        top_countries = list(country_data.keys())[:top_n]
        
        # 指标标准化函数
        def normalize(values):
            min_v, max_v = min(values), max(values)
            if max_v - min_v == 0:
                return [0.5] * len(values)
            return [(v - min_v) / (max_v - min_v) for v in values]
        
        # 收集所有值用于归一化
        all_ttr = [d.get('richness', {}).get('ttr_mean', 0) for d in country_data.values()]
        all_vocab = [d.get('difficulty', {}).get('vocabulary_complexity_index_mean', 0) for d in country_data.values()]
        all_syntactic = [d.get('syntactic', {}).get('syntactic_complexity_index_mean', 0) for d in country_data.values()]
        all_density = [d.get('density', {}).get('information_density_index_mean', 0) for d in country_data.values()]
        all_structure = [d.get('coherence', {}).get('structure_complexity_index_mean', 0) for d in country_data.values()]
        
        categories = ['词汇丰富度', '词汇复杂度', '句法复杂度', '信息密度', '结构复杂度']
        
        fig = go.Figure()
        
        colors = px.colors.qualitative.Set2
        
        for idx, country in enumerate(top_countries):
            data = country_data[country]
            
            ttr = data.get('richness', {}).get('ttr_mean', 0)
            vocab = data.get('difficulty', {}).get('vocabulary_complexity_index_mean', 0)
            syntactic = data.get('syntactic', {}).get('syntactic_complexity_index_mean', 0)
            density = data.get('density', {}).get('information_density_index_mean', 0)
            structure = data.get('coherence', {}).get('structure_complexity_index_mean', 0)
            
            # 归一化
            norm_ttr = normalize(all_ttr)[list(country_data.keys()).index(country)]
            norm_vocab = normalize(all_vocab)[list(country_data.keys()).index(country)]
            norm_syntactic = normalize(all_syntactic)[list(country_data.keys()).index(country)]
            norm_density = normalize(all_density)[list(country_data.keys()).index(country)]
            norm_structure = normalize(all_structure)[list(country_data.keys()).index(country)]
            
            fig.add_trace(go.Scatterpolar(
                r=[norm_ttr, norm_vocab, norm_syntactic, norm_density, norm_structure],
                theta=categories,
                fill='toself',
                name=country,
                line_color=colors[idx % len(colors)],
                opacity=0.7
            ))
        
        fig.update_layout(
            title=f'Top {top_n} 国家/地区语言学特征雷达图<br><sup>(Radar Chart of Linguistic Features - Top {top_n} Countries)</sup>',
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 1])
            ),
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
            height=600,
            paper_bgcolor='#F5FAFF',
            font=dict(color='#1565C0')
        )
        
        fig.write_html(str(self.output_dir / 'country_radar_comparison.html'))
        
        # 排名柱状图
        rankings = country_results['rankings']
        
        for ranking_name, ranking_data in rankings.items():
            top_ranking = ranking_data[:top_n]
            countries = [r[0] for r in top_ranking]
            values = [r[1] for r in top_ranking]
            
            title_map = {
                'vocabulary_richness': ('词汇丰富度排名', 'Vocabulary Richness Ranking'),
                'vocabulary_complexity': ('词汇复杂度排名', 'Vocabulary Complexity Ranking'),
                'syntactic_complexity': ('句法复杂度排名', 'Syntactic Complexity Ranking'),
                'information_density': ('信息密度排名', 'Information Density Ranking'),
                'structure_complexity': ('结构复杂度排名', 'Structure Complexity Ranking')
            }
            
            cn_title, en_title = title_map.get(ranking_name, (ranking_name, ranking_name))
            
            fig_bar = go.Figure()
            fig_bar.add_trace(go.Bar(
                y=countries[::-1],
                x=values[::-1],
                orientation='h',
                marker_color='#1565C0',
                text=[f'{v:.4f}' for v in values[::-1]],
                textposition='outside'
            ))
            
            fig_bar.update_layout(
                title=f'{cn_title}<br><sup>({en_title})</sup>',
                xaxis_title='指标值 Value',
                yaxis_title='国家/地区 Country',
                height=500,
                paper_bgcolor='#F5FAFF',
                font=dict(color='#1565C0'),
                yaxis=dict(tickfont=dict(size=10))
            )
            
            fig_bar.write_html(str(self.output_dir / f'ranking_{ranking_name}.html'))
        
        # 文档数量分布
        doc_counts = {c: d['doc_count'] for c, d in country_data.items()}
        sorted_countries = sorted(doc_counts.items(), key=lambda x: -x[1])[:top_n]
        
        fig_docs = go.Figure()
        fig_docs.add_trace(go.Bar(
            y=[c[0] for c in sorted_countries[::-1]],
            x=[c[1] for c in sorted_countries[::-1]],
            orientation='h',
            marker_color='#D32F2F',
            text=[c[1] for c in sorted_countries[::-1]],
            textposition='outside'
        ))
        
        fig_docs.update_layout(
            title=f'各国/地区文档数量分布 (Top {top_n})<br><sup>(Document Distribution by Country - Top {top_n})</sup>',
            xaxis_title='文档数量 Document Count',
            yaxis_title='国家/地区 Country',
            height=500,
            paper_bgcolor='#F5FAFF',
            font=dict(color='#1565C0')
        )
        
        fig_docs.write_html(str(self.output_dir / 'country_document_distribution.html'))
    
    def create_comparison_heatmap(self, country_results: Dict):
        """创建国别比较热力图"""
        if not PLOTLY_AVAILABLE:
            logger.warning("⚠️ Plotly未安装，跳过可视化生成")
            return
        
        comparison_matrix = country_results['comparison_matrix']
        country_data = country_results['country_data']
        
        if not comparison_matrix or not country_data:
            return
        
        countries = list(country_data.keys())
        
        # 创建综合指标热力图
        metrics = list(comparison_matrix.keys())
        
        z_data = []
        for metric in metrics:
            row = []
            for country in countries:
                value = comparison_matrix[metric].get(country, 0)
                row.append(value)
            z_data.append(row)
        
        # 归一化
        z_normalized = []
        for row in z_data:
            min_v, max_v = min(row), max(row)
            if max_v - min_v == 0:
                z_normalized.append([0.5] * len(row))
            else:
                z_normalized.append([(v - min_v) / (max_v - min_v) for v in row])
        
        fig = go.Figure(data=go.Heatmap(
            z=z_normalized,
            x=countries,
            y=[m.replace('_mean', '').replace('_', ' ') for m in metrics],
            colorscale='RdYlBu_r',
            showscale=True
        ))
        
        fig.update_layout(
            title='各国/地区语言学特征热力图<br><sup>(Linguistic Features Heatmap by Country)</sup>',
            xaxis_title='国家/地区 Country',
            yaxis_title='指标 Metric',
            height=400,
            paper_bgcolor='#F5FAFF',
            font=dict(color='#1565C0'),
            xaxis=dict(tickangle=45, tickfont=dict(size=9))
        )
        
        fig.write_html(str(self.output_dir / 'country_feature_heatmap.html'))


# ==================== 比较分析报告生成器 ====================
class ComparisonReportGenerator:
    """比较分析报告生成器"""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
    
    def generate_reports(self, time_series_results: Dict, country_results: Dict):
        """生成比较分析报告"""
        self._generate_time_series_report(time_series_results)
        self._generate_country_report(country_results)
        self._generate_summary_report(time_series_results, country_results)
    
    def _generate_time_series_report(self, results: Dict):
        """生成时间序列报告"""
        yearly_data = results['yearly_data']
        trends = results.get('trend_analysis', {})
        
        report = f"""# 时间序列分析报告

## 分析概要

- **分析年份范围**: {min(yearly_data.keys())} - {max(yearly_data.keys())}
- **总年份数**: {len(yearly_data)}
- **分析日期**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 年度文档分布

| 年份 | 文档数量 |
|------|----------|
"""
        for year in sorted(yearly_data.keys()):
            report += f"| {year} | {yearly_data[year]['doc_count']} |\n"
        
        report += "\n## 趋势分析\n\n"
        
        for metric, trend in trends.items():
            direction_emoji = "📈" if trend['direction'] == 'increasing' else "📉"
            report += f"""### {metric}

- **趋势方向**: {direction_emoji} {trend['direction']} 
- **变化率**: {trend['change_pct']:.2f}%
- **相关系数**: {trend['correlation']:.3f}
- **起始值**: {trend['start_value']:.4f}
- **结束值**: {trend['end_value']:.4f}

"""
        
        with open(self.output_dir / 'time_series_analysis_report.md', 'w', encoding='utf-8') as f:
            f.write(report)
    
    def _generate_country_report(self, results: Dict):
        """生成国别比较报告"""
        country_data = results['country_data']
        rankings = results.get('rankings', {})
        
        report = f"""# 国别比较分析报告

## 分析概要

- **分析国家/地区数量**: {len(country_data)}
- **分析日期**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 各国/地区文档数量

| 国家/地区 | 文档数量 | 年份范围 |
|-----------|----------|----------|
"""
        for country, data in country_data.items():
            year_range = data.get('year_range')
            year_str = f"{year_range[0]}-{year_range[1]}" if year_range else "N/A"
            report += f"| {country} | {data['doc_count']} | {year_str} |\n"
        
        report += "\n## 排名榜单\n\n"
        
        for ranking_name, ranking_data in rankings.items():
            title_map = {
                'vocabulary_richness': '词汇丰富度排名',
                'vocabulary_complexity': '词汇复杂度排名',
                'syntactic_complexity': '句法复杂度排名',
                'information_density': '信息密度排名',
                'structure_complexity': '结构复杂度排名'
            }
            
            report += f"### {title_map.get(ranking_name, ranking_name)}\n\n"
            report += "| 排名 | 国家/地区 | 指标值 |\n|------|-----------|--------|\n"
            
            for idx, (country, value) in enumerate(ranking_data[:20], 1):
                report += f"| {idx} | {country} | {value:.4f} |\n"
            report += "\n"
        
        with open(self.output_dir / 'country_comparison_report.md', 'w', encoding='utf-8') as f:
            f.write(report)
    
    def _generate_summary_report(self, time_series: Dict, country: Dict):
        """生成综合摘要报告"""
        report = f"""# 语言学特征比较分析综合报告

## 分析概要

- **分析日期**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 时间序列分析摘要

- **年份范围**: {min(time_series['yearly_data'].keys())} - {max(time_series['yearly_data'].keys())}
- **总年份数**: {len(time_series['yearly_data'])}
- **总文档数**: {sum(d['doc_count'] for d in time_series['yearly_data'].values())}

### 主要趋势发现

"""
        trends = time_series.get('trend_analysis', {})
        for metric, trend in list(trends.items())[:5]:
            direction = "上升" if trend['direction'] == 'increasing' else "下降"
            report += f"- **{metric}**: {direction}趋势，变化率 {trend['change_pct']:.2f}%\n"
        
        report += f"""
## 国别比较分析摘要

- **分析国家/地区数量**: {len(country['country_data'])}
- **最大文档数国家**: {max(country['country_data'].items(), key=lambda x: x[1]['doc_count'])[0]}

### 各指标 Top 5

"""
        rankings = country.get('rankings', {})
        for ranking_name, ranking_data in rankings.items():
            title_map = {
                'vocabulary_richness': '词汇丰富度',
                'vocabulary_complexity': '词汇复杂度',
                'syntactic_complexity': '句法复杂度',
                'information_density': '信息密度',
                'structure_complexity': '结构复杂度'
            }
            top5 = ranking_data[:5]
            report += f"**{title_map.get(ranking_name, ranking_name)}**: "
            report += ", ".join([f"{c}({v:.4f})" for c, v in top5]) + "\n\n"
        
        report += """
## 输出文件

### 时间序列分析
- `time_series_evolution.html`: 时间演变图表
- `document_distribution_by_year.html`: 文档年度分布
- `time_series_analysis_report.md`: 详细分析报告

### 国别比较分析
- `country_radar_comparison.html`: 雷达图比较
- `country_feature_heatmap.html`: 特征热力图
- `country_document_distribution.html`: 文档分布
- `ranking_*.html`: 各指标排名图表
- `country_comparison_report.md`: 详细分析报告
"""
        
        with open(self.output_dir / 'comparison_analysis_summary.md', 'w', encoding='utf-8') as f:
            f.write(report)
    
    def save_results_json(self, time_series_results: Dict, country_results: Dict):
        """保存结果JSON"""
        # 时间序列结果
        ts_output = {
            'yearly_data': {str(k): v for k, v in time_series_results['yearly_data'].items()},
            'trend_analysis': time_series_results.get('trend_analysis', {}),
            'statistics': time_series_results.get('statistics', {})
        }
        
        with open(self.output_dir / 'time_series_results.json', 'w', encoding='utf-8') as f:
            json.dump(ts_output, f, ensure_ascii=False, indent=2, default=str)
        
        # 国别比较结果
        country_output = {
            'country_data': country_results['country_data'],
            'rankings': {k: [(c, float(v)) for c, v in data] for k, data in country_results.get('rankings', {}).items()},
            'statistics': country_results.get('statistics', {})
        }
        
        with open(self.output_dir / 'country_comparison_results.json', 'w', encoding='utf-8') as f:
            json.dump(country_output, f, ensure_ascii=False, indent=2, default=str)


# ==================== 比较分析主函数 ====================
def run_comparison_analysis(countries: List[str] = None, 
                           org_types: List[str] = None,
                           doc_types: List[str] = None,
                           data_sources: List[str] = None,
                           min_docs_per_country: int = 5):
    """运行比较分析（时间序列 + 国别比较）
    
    Args:
        countries: 国家/地区筛选列表
        org_types: 组织类型筛选列表
        doc_types: 文档类型筛选列表
        data_sources: 数据源筛选列表
        min_docs_per_country: 国别比较的最小文档数阈值
    
    Returns:
        分析结果字典
    """
    global progress
    progress = ProgressTracker()
    
    logger.info("═" * 80)
    logger.info("📊 文本语言学特征比较分析系统启动")
    logger.info(f"📅 开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("═" * 80)
    
    # ========== 阶段1: 初始化配置 ==========
    progress.start_stage(0)
    config = LinguisticAnalysisConfig.__new__(LinguisticAnalysisConfig)
    config._init_paths()
    config._init_filters()
    config._init_linguistic_settings()
    
    # 应用筛选条件
    if countries:
        config.filter_countries = countries
        logger.info(f"🌍 国家筛选: {countries}")
    if org_types:
        config.filter_org_types = org_types
        logger.info(f"🏢 组织类型筛选: {org_types}")
    if doc_types:
        config.filter_doc_types = doc_types
        logger.info(f"📄 文档类型筛选: {doc_types}")
    if data_sources:
        config.filter_data_sources = data_sources
        logger.info(f"📂 数据源筛选: {data_sources}")
    
    # 设置比较分析输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config.output_dir = config.base_output_dir / f"comparison_analysis_{timestamp}"
    config.output_dir.mkdir(exist_ok=True, parents=True)
    
    # 初始化缓存
    cache_manager = LinguisticCacheManager(config.cache_dir)
    progress.end_stage()
    
    # ========== 阶段2: 加载停用词 ==========
    progress.start_stage(1)
    stopwords_loader = StopwordsLoader(config.stopwords_paths)
    stop_words = stopwords_loader.load_all_stopwords()
    progress.end_stage()
    
    # ========== 阶段3: 加载元数据 ==========
    progress.start_stage(2)
    metadata_loader = MetadataLoader(config.metadata_dir)
    progress.end_stage()
    
    # ========== 阶段4: 加载文档 ==========
    progress.start_stage(3)
    doc_loader = DocumentLoader(
        config.agora_fulltext_dir,
        config.original_data_dir,
        metadata_loader,
        config
    )
    documents = doc_loader.load_documents()
    
    if not documents:
        logger.error("❌ 没有找到有效文档")
        return
    progress.end_stage()
    
    # ========== 阶段5-9: 执行基础分析 ==========
    results = {}
    analyzers = [
        ('词汇丰富度分析', VocabularyRichnessAnalyzer, 'richness_results'),
        ('词汇难度分析', VocabularyDifficultyAnalyzer, 'difficulty_results'),
        ('句法复杂度分析', SyntacticComplexityAnalyzer, 'syntactic_results'),
        ('信息密度分析', InformationDensityAnalyzer, 'density_results'),
        ('结构连贯性分析', StructureCoherenceAnalyzer, 'coherence_results')
    ]
    
    for stage_idx, (name, AnalyzerClass, result_key) in enumerate(analyzers, start=4):
        progress.start_stage(stage_idx)
        analyzer = AnalyzerClass(config, stop_words)
        analyzer.cache_manager = cache_manager
        results[result_key] = analyzer.analyze_batch(documents)
        progress.end_stage()
    
    # ========== 阶段10: 时间序列分析 ==========
    progress.start_stage(9)
    logger.info("📈 执行时间序列分析...")
    time_series_analyzer = TimeSeriesAnalyzer(config, metadata_loader)
    time_series_results = time_series_analyzer.analyze_time_series(
        documents,
        results['richness_results'],
        results['difficulty_results'],
        results['syntactic_results'],
        results['density_results'],
        results['coherence_results']
    )
    progress.end_stage()
    
    # ========== 阶段11: 国别比较分析 ==========
    progress.start_stage(10)
    logger.info("🌍 执行国别比较分析...")
    country_analyzer = CountryComparisonAnalyzer(config, metadata_loader)
    country_results = country_analyzer.analyze_countries(
        documents,
        results['richness_results'],
        results['difficulty_results'],
        results['syntactic_results'],
        results['density_results'],
        results['coherence_results'],
        min_docs=min_docs_per_country
    )
    progress.end_stage()
    
    # ========== 阶段12: 生成可视化 ==========
    progress.start_stage(11)
    logger.info("📊 生成可视化图表...")
    visualizer = ComparisonVisualizer(config.output_dir)
    visualizer.create_time_series_plots(time_series_results)
    visualizer.create_country_comparison_plots(country_results)
    visualizer.create_comparison_heatmap(country_results)
    progress.end_stage()
    
    # ========== 阶段13: 生成报告 ==========
    progress.start_stage(12)
    logger.info("📝 生成分析报告...")
    report_generator = ComparisonReportGenerator(config.output_dir)
    report_generator.generate_reports(time_series_results, country_results)
    report_generator.save_results_json(time_series_results, country_results)
    progress.end_stage()
    
    # 输出缓存统计
    cache_stats = cache_manager.get_stats()
    logger.info(f"💾 缓存统计: 命中率 {cache_stats['hit_rate']:.1%}")
    
    logger.info(f"\n📁 输出目录: {config.output_dir}")
    
    progress.finish()
    
    return {
        'config': config,
        'time_series_results': time_series_results,
        'country_results': country_results,
        'base_results': results
    }


def run_time_series_analysis(countries: List[str] = None, **kwargs):
    """仅运行时间序列分析
    
    Args:
        countries: 国家/地区筛选列表
        **kwargs: 其他筛选条件
    """
    return run_comparison_analysis(countries=countries, **kwargs)


def run_country_comparison(countries: List[str] = None, min_docs: int = 5, **kwargs):
    """仅运行国别比较分析
    
    Args:
        countries: 国家/地区筛选列表（可选，不指定则比较所有国家）
        min_docs: 最小文档数阈值
        **kwargs: 其他筛选条件
    """
    return run_comparison_analysis(countries=countries, min_docs_per_country=min_docs, **kwargs)


if __name__ == "__main__":
    # 默认运行完整比较分析
    run_comparison_analysis()
    
    # 示例用法：
    # run_comparison_analysis(countries=['美国', '中国', '欧盟'])
    # run_time_series_analysis(org_types=['government'])
    # run_country_comparison(min_docs=10)
