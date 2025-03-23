#!/usr/bin/env python3
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
from collections import defaultdict
import json
import csv
import os
from pathlib import Path


class CSRankingsDashboard:
    def __init__(self, root_dir=None):
        """
        Initialize the dashboard with a configurable root directory

        Args:
            root_dir: Path to CSRankings data directory. If None, uses the directory of this script.
        """
        if root_dir is None:
            # Default to the directory where this script is located
            self.root_dir = Path(__file__).parent.absolute()
        else:
            self.root_dir = Path(root_dir)

        self.load_data()
        self.conf_aliases = self.get_conference_aliases()

    def load_data(self):
        """Load necessary data files"""
        # Load article data
        articles_path = self.root_dir / "articles.json"
        try:
            with open(articles_path, "r") as f:
                self.articles = json.load(f)
        except FileNotFoundError:
            st.error(f"Could not find articles.json at {articles_path}")
            self.articles = []

        # Load non-US institution information
        self.non_us_institutions = set()
        country_info_path = self.root_dir / "country-info.csv"
        if country_info_path.exists():
            with open(country_info_path, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row["countryabbrv"].lower() != "us":
                        self.non_us_institutions.add(row["institution"])
        else:
            st.warning(
                f"Country info file not found at {country_info_path}. All institutions will be treated as US institutions.")

        # Load author-institution mapping
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
            st.error(f"Could not find faculty-affiliations.csv at {faculty_path}")

        # Default assumption: institutions not in the non-US list are US institutions
        self.us_institutions = all_institutions - self.non_us_institutions

    def get_conference_aliases(self):
        """返回会议别名到标准名称的映射"""
        # 基于CSRankings.py的定义构建会议别名映射
        return {
            # VLDB别名
            "Proc. VLDB Endow.": "VLDB",
            "PVLDB": "VLDB",

            # ACL别名
            "ACL (1)": "ACL",
            "ACL (2)": "ACL",
            "ACL/IJCNLP": "ACL",
            "ACL/IJCNLP (1)": "ACL",
            "ACL/IJCNLP (2)": "ACL",
            "COLING-ACL": "ACL",

            # CAV别名
            "CAV (1)": "CAV",
            "CAV (2)": "CAV",
            "CAV (3)": "CAV",

            # CRYPTO别名
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

            # ECCV别名
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

            # EMSOFT别名
            "ACM Trans. Embedded Comput. Syst.": "EMSOFT",
            "ACM Trans. Embed. Comput. Syst.": "EMSOFT",
            "IEEE Trans. Comput. Aided Des. Integr. Circuits Syst.": "EMSOFT",

            # EUROCRYPT别名
            "EUROCRYPT (1)": "EUROCRYPT",
            "EUROCRYPT (2)": "EUROCRYPT",
            "EUROCRYPT (3)": "EUROCRYPT",
            "EUROCRYPT (4)": "EUROCRYPT",
            "EUROCRYPT (5)": "EUROCRYPT",

            # Eurographics别名
            "Comput. Graph. Forum": "Eurographics",
            "EUROGRAPHICS": "Eurographics",

            # FSE别名
            "SIGSOFT FSE": "FSE",
            "ESEC/SIGSOFT FSE": "FSE",
            "Proc. ACM Softw. Eng.": "FSE",

            # IEEE VR别名
            "VR": "IEEE VR",

            # ISMB别名
            "Bioinformatics": "ISMB",
            "Bioinform.": "ISMB",
            "ISMB/ECCB (Supplement of Bioinformatics)": "ISMB",
            "Bioinformatics [ISMB/ECCB]": "ISMB",
            "ISMB (Supplement of Bioinformatics)": "ISMB",

            # NAACL别名
            "HLT-NAACL": "NAACL",
            "NAACL-HLT": "NAACL",
            "NAACL-HLT (1)": "NAACL",

            # OOPSLA别名
            "OOPSLA/ECOOP": "OOPSLA",
            "OOPSLA1": "OOPSLA",
            "OOPSLA2": "OOPSLA",
            "PACMPL": "OOPSLA",  # 这个可能需要更精确的处理
            "Proc. ACM Program. Lang.": "OOPSLA",  # 需要更精确的处理

            # Oakland别名
            "IEEE Symposium on Security and Privacy": "Oakland",
            "SP": "Oakland",
            "S&P": "Oakland",

            # PETS别名
            "PoPETs": "PETS",
            "Privacy Enhancing Technologies": "PETS",
            "Proc. Priv. Enhancing Technol.": "PETS",

            # RSS别名
            "Robotics: Science and Systems": "RSS",

            # SIGCSE别名
            "SIGCSE (1)": "SIGCSE",

            # SIGMETRICS别名
            "SIGMETRICS/Performance": "SIGMETRICS",
            "POMACS": "SIGMETRICS",
            "Proc. ACM Meas. Anal. Comput. Syst.": "SIGMETRICS",

            # SIGMOD别名
            "SIGMOD Conference": "SIGMOD",
            "Proc. ACM Manag. Data": "SIGMOD",

            # USENIX Security别名
            "USENIX Security Symposium": "USENIX Security",

            # Ubicomp别名
            "UbiComp": "Ubicomp",
            "IMWUT": "Ubicomp",
            "Pervasive": "Ubicomp",
            "Proc. ACM Interact. Mob. Wearable Ubiquitous Technol.": "Ubicomp",

            # VIS别名
            "IEEE Visualization": "VIS",
            "IEEE Trans. Vis. Comput. Graph.": "VIS"
        }

    def get_areas_and_conferences(self):
        """Get all research areas and conference details"""
        areas = {
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
        return areas

    def create_streamlit_app(self):
        """Create Streamlit application"""
        st.set_page_config(page_title="Academic Analysis Dashboard", layout="wide")
        st.title("Academic Publications Analysis Dashboard")

        # Sidebar configuration
        st.sidebar.header("Configuration")

        # Get areas and conferences information
        areas_info = self.get_areas_and_conferences()

        # Analysis type selection
        analysis_type = st.sidebar.radio(
            "Select analysis type",
            ["Top 100 Institutions", "Top 100 Scholars"],
            help="Select the type of entity you want to analyze"
        )

        # Research area multi-select
        selected_top_level_areas = st.sidebar.multiselect(
            "Select research areas",
            list(areas_info.keys()),
            help="Select research areas you're interested in"
        )

        # Update available conferences based on selected areas
        all_available_conferences = []
        if selected_top_level_areas:
            for area in selected_top_level_areas:
                all_available_conferences.extend(areas_info[area]["conferences"])

        selected_conferences = st.sidebar.multiselect(
            "Select conferences",
            list(set(all_available_conferences)),
            help="Select specific conferences you want to analyze"
        )

        # Year selection
        start_year, end_year = st.sidebar.slider(
            "Select year range",
            min_value=2010,
            max_value=2025,
            value=(2020, 2025),
            help="Select the year range for your analysis"
        )

        # Execute analysis button
        if st.sidebar.button("Start Analysis", type="primary"):
            # Filter articles
            filtered_articles = self.filter_articles(
                selected_conferences,
                start_year,
                end_year
            )

            if not filtered_articles:
                st.warning("No articles found matching your criteria. We'll still try to display any available data.")
                # Continue execution even if no articles found
                # This allows showing partial or empty results instead of stopping

            # Choose analysis method based on type
            if analysis_type == "Top 100 Institutions":
                # Analyze top institutions - 不限制为50个，显示所有有发表的机构
                top_institutions, inst_yearly_counts, inst_top_authors = self.analyze_top_institutions(
                    filtered_articles, top_n=None)

                st.write(f"发现 {len(top_institutions)} 个具有符合条件发表的机构")

                # Create dataframe for display - even if empty, we'll create a structure
                inst_data = []
                # 限制最多显示100个机构
                display_institutions = top_institutions[:100]

                for rank, (inst, total) in enumerate(display_institutions, 1):
                    row = {
                        "Rank": rank,
                        "Institution": inst,
                        "Total Papers": total
                    }

                    # Add yearly paper counts
                    for year in range(start_year, end_year + 1):
                        row[str(year)] = inst_yearly_counts[inst].get(year, 0)

                    # Add Top 10 authors
                    top_10_authors = inst_top_authors[inst]
                    row["Top 10 Authors"] = ", ".join([f"{author}({count})" for author, count in top_10_authors])

                    inst_data.append(row)

                # Even if we have no data, create an empty DataFrame with proper columns
                if not inst_data:
                    columns = ["Rank", "Institution", "Total Papers", "Top 10 Authors"]
                    columns.extend([str(year) for year in range(start_year, end_year + 1)])
                    df = pd.DataFrame(columns=columns)
                    st.info("No institutions found with papers matching your criteria.")
                else:
                    df = pd.DataFrame(inst_data)

                    # Display results
                    st.header("Institution Analysis Results")

                    # Institution paper count table
                    with st.expander(f"Institutions with Publications ({len(df)} found)", expanded=True):
                        st.dataframe(df, use_container_width=True)

                    # Only display charts if we have data
                    if not df.empty:
                        # Institution paper count bar chart
                        fig = px.bar(
                            df,
                            x='Institution',
                            y='Total Papers',
                            title=f"{start_year}-{end_year} US Institutions with Publications in Selected Conferences",
                            labels={'Total Papers': 'Paper Count', 'Institution': 'Institution Name'}
                        )
                        st.plotly_chart(fig, use_container_width=True)

                        # Yearly trends - only top 5 institutions or all if less than 5
                        top_n = min(5, len(df))
                        if top_n > 0:
                            top5_df = df.head(top_n)
                            # 确保所有年份都有数据
                            year_columns = [str(year) for year in range(start_year, end_year + 1)]
                            yearly_trend = top5_df[year_columns]

                            # 为每个机构创建每年的数据
                            trend_data = []
                            for idx, row in top5_df.iterrows():
                                institution = row['Institution']
                                for year in year_columns:
                                    trend_data.append({
                                        'Institution': institution,
                                        'Year': year,
                                        'Papers': row[year]
                                    })

                            trend_df = pd.DataFrame(trend_data)

                            trend_fig = px.line(
                                trend_df,
                                x='Year',
                                y='Papers',
                                color='Institution',
                                title=f"{start_year}-{end_year} Top {top_n} Institutions Yearly Paper Trends",
                                markers=True
                            )
                            # Improve line visibility and ensure x-axis shows all years
                            trend_fig.update_traces(line=dict(width=3))
                            trend_fig.update_xaxes(tickvals=year_columns)
                            st.plotly_chart(trend_fig, use_container_width=True)
                        else:
                            st.info("No trend data available for visualization.")

            else:  # Top 100 Scholars
                # Analyze top authors - 不限制为100个，显示所有有发表的学者
                top_authors, author_yearly_counts = self.analyze_top_authors(filtered_articles, top_n=None)

                #st.write(f"发现 {len(top_authors)} 位具有符合条件发表的学者")

                # Create dataframe for display - even if empty, we'll create a structure
                author_data = []
                # 限制最多显示100个学者
                display_authors = top_authors[:100]

                for rank, (author, total) in enumerate(display_authors, 1):
                    row = {
                        "Rank": rank,
                        "Author": author,
                        "Institution": self.author_institutions.get(author, "Unknown"),
                        "Total Papers": total
                    }

                    # Add yearly paper counts
                    for year in range(start_year, end_year + 1):
                        row[str(year)] = author_yearly_counts[author].get(year, 0)

                    author_data.append(row)

                # Even if we have no data, create an empty DataFrame with proper columns
                if not author_data:
                    columns = ["Rank", "Author", "Institution", "Total Papers"]
                    columns.extend([str(year) for year in range(start_year, end_year + 1)])
                    df = pd.DataFrame(columns=columns)
                    st.info("No authors found with papers matching your criteria.")
                else:
                    df = pd.DataFrame(author_data)

                    # Display results
                    st.header("Scholar Analysis Results")

                    # Author paper count table
                    with st.expander(f"Scholars with Publications ({len(df)} found)", expanded=True):
                        st.dataframe(df, use_container_width=True)

                    # Only display charts if we have data
                    if not df.empty:
                        # Author paper count bar chart
                        fig = px.bar(
                            df,
                            x='Author',
                            y='Total Papers',
                            title=f"{start_year}-{end_year} US Authors with Publications in Selected Conferences",
                            labels={'Total Papers': 'Paper Count', 'Author': 'Author Name'}
                        )
                        st.plotly_chart(fig, use_container_width=True)

                        # Yearly trends - only top 5 authors or all if less than 5
                        top_n = min(5, len(df))
                        if top_n > 0:
                            top5_df = df.head(top_n)
                            # 确保所有年份都有数据
                            year_columns = [str(year) for year in range(start_year, end_year + 1)]

                            # 为每个作者创建每年的数据
                            trend_data = []
                            for idx, row in top5_df.iterrows():
                                author = row['Author']
                                for year in year_columns:
                                    trend_data.append({
                                        'Author': author,
                                        'Year': year,
                                        'Papers': row[year]
                                    })

                            trend_df = pd.DataFrame(trend_data)

                            trend_fig = px.line(
                                trend_df,
                                x='Year',
                                y='Papers',
                                color='Author',
                                title=f"{start_year}-{end_year} Top {top_n} Scholars Yearly Paper Trends",
                                markers=True
                            )
                            # Improve line visibility and ensure x-axis shows all years
                            trend_fig.update_traces(line=dict(width=3))
                            trend_fig.update_xaxes(tickvals=year_columns)
                            st.plotly_chart(trend_fig, use_container_width=True)
                        else:
                            st.info("No trend data available for visualization.")

    def filter_articles(self, selected_conferences, start_year, end_year):
        """Filter articles based on criteria"""
        filtered_articles = []
        conference_counts = defaultdict(int)

        # 如果有选中的会议，将其添加到映射中
        selected_confs_with_aliases = set()
        alias_to_canonical = {}

        if selected_conferences:
            for conf in selected_conferences:
                selected_confs_with_aliases.add(conf)
                # 添加这个会议的所有别名
                for alias, canonical in self.conf_aliases.items():
                    if canonical == conf:
                        selected_confs_with_aliases.add(alias)
                        alias_to_canonical[alias] = canonical

        for article in self.articles:
            conf_orig = article.get("conf")
            year = int(article.get("year", 0))

            # 应用会议别名转换
            conf = conf_orig
            if conf in self.conf_aliases:
                conf = self.conf_aliases[conf]
                # 在处理之前保留原始名称
                article["original_conf"] = conf_orig

            # 如果没有选择会议或者会议在选择列表中，并且年份在范围内
            if (not selected_conferences or conf in selected_confs_with_aliases) and \
                    start_year <= year <= end_year:
                filtered_articles.append(article)

                # 统计时使用规范化后的会议名称
                conf_for_counting = conf
                if conf_orig in alias_to_canonical:
                    conf_for_counting = alias_to_canonical[conf_orig]

                conference_counts[conf_for_counting] += 1

        # 打印匹配的文章数量和会议分布
        st.write(f"找到 {len(filtered_articles)} 篇符合条件的文章")
        if filtered_articles:
            st.write("会议分布:")
            # 按数量排序展示
            sorted_confs = sorted(conference_counts.items(), key=lambda x: x[1], reverse=True)
            for conf, count in sorted_confs:
                st.write(f"  - {conf}: {count} 篇")

            # 显示别名使用情况
            alias_usage = defaultdict(int)
            for article in filtered_articles:
                if "original_conf" in article and article["original_conf"] != article.get("conf"):
                    alias_usage[f"{article['original_conf']} -> {article.get('conf')}"] += 1

            if alias_usage:
                st.write("会议别名使用情况:")
                for alias_map, count in alias_usage.items():
                    st.write(f"  - {alias_map}: {count} 篇")

        return filtered_articles

    def analyze_top_authors(self, filtered_articles, top_n=100):
        """Analyze top authors"""
        # 如果top_n是None，则显示所有有发表的作者
        author_counts = defaultdict(int)
        author_yearly_counts = defaultdict(lambda: defaultdict(int))

        for article in filtered_articles:
            author = article.get("name")
            year = int(article.get("year", 0))

            # Only count US institution authors
            if author in self.author_institutions and \
                    self.author_institutions[author] in self.us_institutions:
                author_counts[author] += 1
                author_yearly_counts[author][year] += 1

        # Sort by total paper count
        sorted_authors = sorted(author_counts.items(), key=lambda x: x[1], reverse=True)

        # 如果top_n是None，显示所有结果；否则，截取前top_n个
        top_authors = sorted_authors if top_n is None else sorted_authors[:top_n]

        return top_authors, author_yearly_counts

    def analyze_top_institutions(self, filtered_articles, top_n=50):
        """Analyze top institutions"""
        # 如果top_n是None，则显示所有有发表的机构
        institution_counts = defaultdict(int)
        institution_yearly_counts = defaultdict(lambda: defaultdict(int))
        institution_top_authors = defaultdict(lambda: defaultdict(int))

        for article in filtered_articles:
            author = article.get("name")
            year = int(article.get("year", 0))

            # Only count US institutions
            if author in self.author_institutions:
                institution = self.author_institutions[author]
                if institution in self.us_institutions:
                    institution_counts[institution] += 1
                    institution_yearly_counts[institution][year] += 1
                    institution_top_authors[institution][author] += 1

        # Sort institutions by total paper count
        sorted_institutions = sorted(institution_counts.items(), key=lambda x: x[1], reverse=True)

        # 如果top_n是None，显示所有结果；否则，截取前top_n个
        top_institutions = sorted_institutions if top_n is None else sorted_institutions[:top_n]

        # Get Top 10 authors for each institution
        institution_top10_authors = {}
        for inst in institution_counts:
            # Sort authors by paper count
            sorted_authors = sorted(institution_top_authors[inst].items(), key=lambda x: x[1], reverse=True)[:10]
            institution_top10_authors[inst] = sorted_authors

        return top_institutions, institution_yearly_counts, institution_top10_authors


def main():
    # Use current directory by default, can be overridden with environment variables
    data_dir = os.environ.get("CSRANKINGS_DATA_DIR", None)
    dashboard = CSRankingsDashboard(data_dir)
    dashboard.create_streamlit_app()


if __name__ == "__main__":
    main()
