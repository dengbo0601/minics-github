#!/usr/bin/env python3
import streamlit as st
import pandas as pd
import plotly.express as px
import json
import csv
import os
from collections import defaultdict
from pathlib import Path
import copy


class CSRankingsDashboard:
    def __init__(self, root_dir=None):
        """初始化仪表板，可配置根目录"""
        if root_dir is None:
            self.root_dir = Path(__file__).parent.absolute()
        else:
            self.root_dir = Path(root_dir)

        # 初始化并加载数据
        self.load_data()
        self.conf_aliases = self.get_conference_aliases()
        self.areas_info = self.get_areas_and_conferences()

    def load_data(self):
        """加载必要的数据文件"""
        # 加载文章数据
        articles_path = self.root_dir / "articles.json"
        try:
            with open(articles_path, "r") as f:
                self.articles = json.load(f)
        except FileNotFoundError:
            st.error(f"找不到articles.json文件: {articles_path}")
            self.articles = []

        # 加载非美国机构信息
        self.non_us_institutions = set()
        country_info_path = self.root_dir / "country-info.csv"
        if country_info_path.exists():
            with open(country_info_path, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row["countryabbrv"].lower() != "us":
                        self.non_us_institutions.add(row["institution"])
        else:
            st.warning(f"找不到国家信息文件: {country_info_path}。所有机构将被视为美国机构。")

        # 加载作者-机构映射
        self.author_institutions = {}
        all_institutions = set()
        faculty_path = self.root_dir / "faculty-affiliations.csv"
        try:
            with open(faculty_path, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    self.author_institutions[row["name"]] = row["affiliation"]
                    all_institutions.add(row["affiliation"])
        except FileNotFoundError:
            st.error(f"找不到faculty-affiliations.csv文件: {faculty_path}")
            all_institutions = set()

        # 默认假设：非美国机构列表中未包含的机构是美国机构
        self.us_institutions = all_institutions - self.non_us_institutions

    def get_conference_aliases(self):
        """返回会议别名到标准名称的映射"""
        return {
            # VLDB aliases
            "Proc. VLDB Endow.": "VLDB",
            "PVLDB": "VLDB",

            # ACL aliases
            "ACL (1)": "ACL",
            "ACL (2)": "ACL",
            "ACL/IJCNLP": "ACL",
            "ACL/IJCNLP (1)": "ACL",
            "ACL/IJCNLP (2)": "ACL",
            "COLING-ACL": "ACL",

            # CAV aliases
            "CAV (1)": "CAV",
            "CAV (2)": "CAV",
            "CAV (3)": "CAV",

            # CRYPTO aliases
            "CRYPTO (1)": "CRYPTO",
            "CRYPTO (2)": "CRYPTO",
            "CRYPTO (3)": "CRYPTO",
            "CRYPTO (4)": "CRYPTO",
            "CRYPTO (5)": "CRYPTO",
            "CRYPTO (6)": "CRYPTO",
            "CRYPTO (7)": "CRYPTO",
            "CRYPTO (8)": "CRYPTO",
            "CRYPTO (9)": "CRYPTO",
            "CRYPTO (10)": "CRYPTO",

            # ECCV aliases
            "ECCV (1)": "ECCV",
            "ECCV (2)": "ECCV",
            "ECCV (3)": "ECCV",
            "ECCV (4)": "ECCV",
            "ECCV (5)": "ECCV",
            "ECCV (6)": "ECCV",
            "ECCV (7)": "ECCV",
            "ECCV (8)": "ECCV",
            "ECCV (9)": "ECCV",
            "ECCV (10)": "ECCV",
            "ECCV (11)": "ECCV",
            "ECCV (12)": "ECCV",
            "ECCV (13)": "ECCV",
            "ECCV (14)": "ECCV",
            "ECCV (15)": "ECCV",
            "ECCV (16)": "ECCV",
            "ECCV (17)": "ECCV",
            "ECCV (18)": "ECCV",
            "ECCV (19)": "ECCV",
            "ECCV (20)": "ECCV",
            "ECCV (21)": "ECCV",
            "ECCV (22)": "ECCV",
            "ECCV (23)": "ECCV",
            "ECCV (24)": "ECCV",
            "ECCV (25)": "ECCV",
            "ECCV (26)": "ECCV",
            "ECCV (27)": "ECCV",
            "ECCV (28)": "ECCV",
            "ECCV (29)": "ECCV",
            "ECCV (30)": "ECCV",
            "ECCV (31)": "ECCV",
            "ECCV (32)": "ECCV",
            "ECCV (33)": "ECCV",
            "ECCV (34)": "ECCV",
            "ECCV (35)": "ECCV",
            "ECCV (36)": "ECCV",
            "ECCV (37)": "ECCV",
            "ECCV (38)": "ECCV",
            "ECCV (39)": "ECCV",

            # EMSOFT aliases
            "ACM Trans. Embedded Comput. Syst.": "EMSOFT",
            "ACM Trans. Embed. Comput. Syst.": "EMSOFT",
            "IEEE Trans. Comput. Aided Des. Integr. Circuits Syst.": "EMSOFT",

            # EUROCRYPT aliases
            "EUROCRYPT (1)": "EUROCRYPT",
            "EUROCRYPT (2)": "EUROCRYPT",
            "EUROCRYPT (3)": "EUROCRYPT",
            "EUROCRYPT (4)": "EUROCRYPT",
            "EUROCRYPT (5)": "EUROCRYPT",

            # Eurographics aliases
            "Comput. Graph. Forum": "Eurographics",
            "EUROGRAPHICS": "Eurographics",

            # FSE aliases
            "SIGSOFT FSE": "FSE",
            "ESEC/SIGSOFT FSE": "FSE",
            "Proc. ACM Softw. Eng.": "FSE",

            # IEEE VR aliases
            "VR": "IEEE VR",

            # ISMB aliases
            "Bioinformatics": "ISMB",
            "Bioinform.": "ISMB",
            "ISMB/ECCB (Supplement of Bioinformatics)": "ISMB",
            "Bioinformatics [ISMB/ECCB]": "ISMB",
            "ISMB (Supplement of Bioinformatics)": "ISMB",

            # NAACL aliases
            "HLT-NAACL": "NAACL",
            "NAACL-HLT": "NAACL",
            "NAACL-HLT (1)": "NAACL",

            # OOPSLA aliases
            "OOPSLA/ECOOP": "OOPSLA",
            "OOPSLA1": "OOPSLA",
            "OOPSLA2": "OOPSLA",
            "PACMPL": "OOPSLA",  # This may need more precise handling
            "Proc. ACM Program. Lang.": "OOPSLA",  # Needs more precise handling

            # Oakland aliases
            "IEEE Symposium on Security and Privacy": "Oakland",
            "SP": "Oakland",
            "S&P": "Oakland",

            # PETS aliases
            "PoPETs": "PETS",
            "Privacy Enhancing Technologies": "PETS",
            "Proc. Priv. Enhancing Technol.": "PETS",

            # RSS aliases
            "Robotics: Science and Systems": "RSS",

            # SIGCSE aliases
            "SIGCSE (1)": "SIGCSE",

            # SIGMETRICS aliases
            "SIGMETRICS/Performance": "SIGMETRICS",
            "POMACS": "SIGMETRICS",
            "Proc. ACM Meas. Anal. Comput. Syst.": "SIGMETRICS",

            # SIGMOD aliases
            "SIGMOD Conference": "SIGMOD",
            "Proc. ACM Manag. Data": "SIGMOD",

            # USENIX Security aliases
            "USENIX Security Symposium": "USENIX Security",

            # Ubicomp aliases
            "UbiComp": "Ubicomp",
            "IMWUT": "Ubicomp",
            "Pervasive": "Ubicomp",
            "Proc. ACM Interact. Mob. Wearable Ubiquitous Technol.": "Ubicomp",

            # VIS aliases
            "IEEE Visualization": "VIS",
            "IEEE Trans. Vis. Comput. Graph.": "VIS"
        }

    def get_areas_and_conferences(self):
        """获取所有研究领域和会议详细信息"""
        return {
            "AI": {
                "areas": ["AI", "Machine Learning", "NLP"],
                "conferences": ["NeurIPS", "ICML", "ICLR", "AAAI", "ACL", "EMNLP", "NAACL", "IJCAI"]
            },
            "Computer Vision": {
                "areas": ["Computer Vision"],
                "conferences": ["CVPR", "ICCV", "ECCV"]
            },
            "Computer Graphics": {
                "areas": ["Graphics"],
                "conferences": ["SIGGRAPH", "SIGGRAPH Asia", "Eurographics"]
            },
            "Computer Architecture": {
                "areas": ["Computer Architecture"],
                "conferences": ["ISCA", "MICRO", "HPCA", "SC", "HPDC", "ICS"]
            },
            "Computer Systems": {
                "areas": ["Operating Systems", "Networks", "Storage"],
                "conferences": ["ASPLOS", "SOSP", "OSDI", "NSDI", "SIGCOMM", "FAST", "USENIX ATC", "EuroSys"]
            },
            "Databases": {
                "areas": ["Databases"],
                "conferences": ["SIGMOD", "VLDB", "ICDE", "PODS"]
            },
            "Programming Languages": {
                "areas": ["Programming Languages"],
                "conferences": ["POPL", "PLDI", "ICFP", "OOPSLA", "CAV", "LICS"]
            },
            "Software Engineering": {
                "areas": ["Software Engineering"],
                "conferences": ["ICSE", "FSE", "ASE", "ISSTA"]
            },
            "Security & Privacy": {
                "areas": ["Security"],
                "conferences": ["CCS", "Oakland", "USENIX Security", "NDSS", "CRYPTO", "EUROCRYPT"]
            },
            "Mobile Computing": {
                "areas": ["Mobile Computing"],
                "conferences": ["MobiCom", "MobiSys", "SenSys"]
            },
            "Human-Computer Interaction": {
                "areas": ["Human-Computer Interaction"],
                "conferences": ["CHI", "UIST", "Ubicomp"]
            },
            "Theoretical Computer Science": {
                "areas": ["Theoretical Computer Science"],
                "conferences": ["FOCS", "STOC", "SODA"]
            },
            "EDA": {
                "areas": ["Electronic Design Automation"],
                "conferences": ["DAC", "ICCAD"]
            },
            "Robotics": {
                "areas": ["Robotics"],
                "conferences": ["ICRA", "IROS", "RSS"]
            },
            "Embedded Systems": {
                "areas": ["Embedded Systems"],
                "conferences": ["EMSOFT", "RTAS", "RTSS"]
            },
            "Visualization": {
                "areas": ["Visualization"],
                "conferences": ["VIS", "IEEE VR"]
            },
            "Web&Information Retrieval": {
                "areas": ["Information Retrieval"],
                "conferences": ["SIGIR", "WWW", "CIKM"]
            },
            "Computational Biology": {
                "areas": ["Bioinformatics"],
                "conferences": ["ISMB", "RECOMB"]
            },
            "E-commerce": {
                "areas": ["E-commerce"],
                "conferences": ["EC", "WINE"]
            },
            "Computer Science Education": {
                "areas": ["CS Education"],
                "conferences": ["SIGCSE"]
            }
        }

    def run_streamlit_app(self):
        """创建Streamlit应用，采用自上而下的布局以获得更好的移动体验"""
        # 设置页面配置
        st.set_page_config(page_title="学术发表分析仪表板", layout="wide")

        # 应用标题
        st.title("学术出版物分析仪表板")

        # 初始化会话状态变量，如果尚未完成
        if 'last_run_id' not in st.session_state:
            st.session_state.last_run_id = 0

        # 使用表单收集所有输入并一次提交
        with st.form("analysis_form"):
            st.subheader("配置分析参数")

            # 使用列组织配置选项
            col1, col2 = st.columns(2)

            with col1:
                # 分析类型选择
                analysis_type = st.radio(
                    "选择分析类型",
                    ["Top 100 Scholars", "Top 100 Institutions"],
                    index=0
                )

                # 年份选择
                start_year, end_year = st.slider(
                    "选择年份范围",
                    min_value=2010,
                    max_value=2025,
                    value=(2020, 2025)
                )

            with col2:
                # 研究领域多选
                selected_areas = st.multiselect(
                    "选择研究领域",
                    list(self.areas_info.keys())
                )

                # 根据所选领域更新可用会议
                all_available_conferences = []
                if selected_areas:
                    for area in selected_areas:
                        all_available_conferences.extend(self.areas_info[area]["conferences"])
                    # 去重
                    all_available_conferences = list(set(all_available_conferences))

                # 会议多选
                selected_conferences = st.multiselect(
                    "选择会议",
                    all_available_conferences
                )

            # 提交按钮 - 居中且更突出
            submit_button = st.form_submit_button(
                "开始分析",
                use_container_width=True,
                type="primary"
            )

        # 当用户提交表单时
        if submit_button:
            # 增加运行ID以跟踪新的分析请求
            st.session_state.last_run_id += 1

            # 在会话状态中保存当前的分析参数
            st.session_state.current_params = {
                "run_id": st.session_state.last_run_id,
                "analysis_type": analysis_type,
                "start_year": start_year,
                "end_year": end_year,
                "selected_conferences": selected_conferences
            }

            # 显示分析状态
            st.success(f"分析已启动（ID: {st.session_state.last_run_id}）")

        # 如果存在当前分析参数，则显示结果
        if 'current_params' in st.session_state:
            params = st.session_state.current_params

            # 水平线分隔配置和结果
            st.markdown("---")

            # 显示当前分析参数
            st.subheader("当前分析参数")
            st.write(f"分析类型: {params['analysis_type']}")
            st.write(f"年份范围: {params['start_year']}-{params['end_year']}")
            st.write(
                f"选定会议: {', '.join(params['selected_conferences']) if params['selected_conferences'] else '所有会议'}")

            # 进行分析
            self.run_analysis(params)

    def run_analysis(self, params):
        """根据参数运行分析"""
        # 过滤文章
        filtered_articles = self.filter_articles(
            params['selected_conferences'],
            params['start_year'],
            params['end_year']
        )

        # 根据分析类型选择分析方法
        if params['analysis_type'] == "Top 100 Institutions":
            self.analyze_institutions(filtered_articles, params)
        else:  # Top 100 Scholars
            self.analyze_scholars(filtered_articles, params)

    def filter_articles(self, selected_conferences, start_year, end_year):
        """根据条件筛选文章"""
        # 深拷贝文章列表，避免修改原始数据
        articles_copy = copy.deepcopy(self.articles)

        # 收集选定会议及其别名
        selected_confs_with_aliases = set()
        alias_to_canonical = {}

        if selected_conferences:
            for conf in selected_conferences:
                selected_confs_with_aliases.add(conf)
                # 添加该会议的所有别名
                for alias, canonical in self.conf_aliases.items():
                    if canonical == conf:
                        selected_confs_with_aliases.add(alias)
                        alias_to_canonical[alias] = canonical

        # 筛选满足条件的文章
        filtered_articles = []
        conference_counts = defaultdict(int)

        for article in articles_copy:
            conf_orig = article.get("conf", "")
            year = int(article.get("year", 0))

            # 应用会议别名转换
            conf = conf_orig
            if conf in self.conf_aliases:
                conf = self.conf_aliases[conf]
                article["normalized_conf"] = conf

            # 如果没有选定会议或会议在选定列表中，且年份在范围内
            if (not selected_conferences or conf in selected_confs_with_aliases) and \
                    start_year <= year <= end_year:
                filtered_articles.append(article)

                # 使用规范化的会议名称进行计数
                conf_for_counting = conf
                conference_counts[conf_for_counting] += 1

        # 输出匹配文章的数量和会议分布
        st.write(f"找到 {len(filtered_articles)} 篇符合条件的文章")
        if filtered_articles:
            st.write("会议分布:")
            # 按计数排序会议
            sorted_confs = sorted(conference_counts.items(), key=lambda x: x[1], reverse=True)
            for conf, count in sorted_confs:
                st.write(f"  - {conf}: {count} 篇论文")

        return filtered_articles

    def analyze_scholars(self, filtered_articles, params):
        """分析顶尖学者"""
        st.subheader("学者分析结果")

        # 计算每位学者的论文数量
        author_counts = defaultdict(int)
        author_yearly_counts = defaultdict(lambda: defaultdict(int))

        for article in filtered_articles:
            author = article.get("name", "")
            year = int(article.get("year", 0))

            # 只计算美国机构的作者
            if author in self.author_institutions and \
                    self.author_institutions[author] in self.us_institutions:
                author_counts[author] += 1
                author_yearly_counts[author][year] += 1

        # 按总论文数排序
        sorted_authors = sorted(author_counts.items(), key=lambda x: x[1], reverse=True)
        st.write(f"找到 {len(sorted_authors)} 位满足条件的学者")

        # 限制显示最多100位学者
        display_authors = sorted_authors[:100]

        if not display_authors:
            st.info("没有找到符合条件的学者。")
            return

        # 创建数据框用于显示
        author_data = []
        for rank, (author, total) in enumerate(display_authors, 1):
            row = {
                "排名": rank,
                "学者": author,
                "机构": self.author_institutions.get(author, "未知"),
                "总论文数": total
            }

            # 添加每年的论文数
            for year in range(params['start_year'], params['end_year'] + 1):
                row[str(year)] = author_yearly_counts[author].get(year, 0)

            author_data.append(row)

        # 创建数据框
        df = pd.DataFrame(author_data)

        # 显示学者论文数量表格
        with st.expander(f"学者发表情况 (找到 {len(df)} 位)", expanded=True):
            st.dataframe(df.set_index('排名'), use_container_width=True)

        # 显示图表
        # 学者论文数量条形图
        fig = px.bar(
            df,
            x='学者',
            y='总论文数',
            title=f"{params['start_year']}-{params['end_year']} 美国学者在选定会议中的发表情况",
            labels={'总论文数': '论文数量', '学者': '学者姓名'}
        )
        st.plotly_chart(fig, use_container_width=True)

        # 每年趋势 - 仅前5位学者或全部（如果少于5位）
        top_n = min(5, len(df))
        if top_n > 0:
            top5_df = df.head(top_n)
            # 确保所有年份的数据
            year_columns = [str(year) for year in range(params['start_year'], params['end_year'] + 1)]

            # 为每位学者创建每年数据
            trend_data = []
            for _, row in top5_df.iterrows():
                author = row['学者']
                for year in year_columns:
                    trend_data.append({
                        '学者': author,
                        '年份': year,
                        '论文数': row[year]
                    })

            trend_df = pd.DataFrame(trend_data)

            trend_fig = px.line(
                trend_df,
                x='年份',
                y='论文数',
                color='学者',
                title=f"{params['start_year']}-{params['end_year']} 前 {top_n} 位学者每年论文趋势",
                markers=True
            )
            # 改进线条可见性并确保x轴显示所有年份
            trend_fig.update_traces(line=dict(width=3))
            trend_fig.update_xaxes(tickvals=year_columns)
            st.plotly_chart(trend_fig, use_container_width=True)

    def analyze_institutions(self, filtered_articles, params):
        """分析顶尖机构"""
        st.subheader("机构分析结果")

        # 计算每个机构的论文数量
        institution_counts = defaultdict(int)
        institution_yearly_counts = defaultdict(lambda: defaultdict(int))
        institution_top_authors = defaultdict(lambda: defaultdict(int))

        for article in filtered_articles:
            author = article.get("name", "")
            year = int(article.get("year", 0))

            # 只计算美国机构
            if author in self.author_institutions:
                institution = self.author_institutions[author]
                if institution in self.us_institutions:
                    institution_counts[institution] += 1
                    institution_yearly_counts[institution][year] += 1
                    institution_top_authors[institution][author] += 1

        # 按总论文数排序
        sorted_institutions = sorted(institution_counts.items(), key=lambda x: x[1], reverse=True)
        st.write(f"找到 {len(sorted_institutions)} 个满足条件的机构")

        # 限制显示最多100个机构
        display_institutions = sorted_institutions[:100]

        if not display_institutions:
            st.info("没有找到符合条件的机构。")
            return

        # 获取每个机构的前10位作者
        institution_top10_authors = {}
        for inst in institution_counts:
            # 按论文数量排序作者
            sorted_authors = sorted(institution_top_authors[inst].items(), key=lambda x: x[1], reverse=True)[:10]
            institution_top10_authors[inst] = sorted_authors

        # 创建数据框用于显示
        inst_data = []
        for rank, (inst, total) in enumerate(display_institutions, 1):
            row = {
                "排名": rank,
                "机构": inst,
                "总论文数": total
            }

            # 添加每年的论文数
            for year in range(params['start_year'], params['end_year'] + 1):
                row[str(year)] = institution_yearly_counts[inst].get(year, 0)

            # 添加前10位作者
            top_10_authors = institution_top10_authors[inst]
            row["前10位作者"] = ", ".join([f"{author}({count})" for author, count in top_10_authors])

            inst_data.append(row)

        # 创建数据框
        df = pd.DataFrame(inst_data)

        # 显示机构论文数量表格
        with st.expander(f"机构发表情况 (找到 {len(df)} 个)", expanded=True):
            st.dataframe(df.set_index('排名'), use_container_width=True)

        # 显示图表
        # 机构论文数量条形图
        fig = px.bar(
            df,
            x='机构',
            y='总论文数',
            title=f"{params['start_year']}-{params['end_year']} 美国机构在选定会议中的发表情况",
            labels={'总论文数': '论文数量', '机构': '机构名称'}
        )
        st.plotly_chart(fig, use_container_width=True)

        # 每年趋势 - 仅前5个机构或全部（如果少于5个）
        top_n = min(5, len(df))
        if top_n > 0:
            top5_df = df.head(top_n)
            # 确保所有年份的数据
            year_columns = [str(year) for year in range(params['start_year'], params['end_year'] + 1)]

            # 为每个机构创建每年数据
            trend_data = []
            for _, row in top5_df.iterrows():
                institution = row['机构']
                for year in year_columns:
                    trend_data.append({
                        '机构': institution,
                        '年份': year,
                        '论文数': row[year]
                    })

            trend_df = pd.DataFrame(trend_data)

            trend_fig = px.line(
                trend_df,
                x='年份',
                y='论文数',
                color='机构',
                title=f"{params['start_year']}-{params['end_year']} 前 {top_n} 个机构每年论文趋势",
                markers=True
            )
            # 改进线条可见性并确保x轴显示所有年份
            trend_fig.update_traces(line=dict(width=3))
            trend_fig.update_xaxes(tickvals=year_columns)
            st.plotly_chart(trend_fig, use_container_width=True)


def main():
    # 默认使用当前目录，可以通过环境变量覆盖
    data_dir = os.environ.get("CSRANKINGS_DATA_DIR", None)
    dashboard = CSRankingsDashboard(data_dir)
    dashboard.run_streamlit_app()


if __name__ == "__main__":
    main()
