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
        """Return a mapping of conference aliases to standard names"""
        # Build conference alias mapping based on CSRankings.py definitions
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
        """Create Streamlit application with a top-down layout for better mobile experience"""
        # Set page configuration
        st.set_page_config(page_title="Academic Analysis Dashboard", layout="wide")
        
        # App title
        st.title("Academic Publications Analysis Dashboard")

        # Initialize session state if not already done
        if 'start_analysis' not in st.session_state:
            st.session_state.start_analysis = False
            
        # Configuration section in a collapsible container at the top
        with st.expander("📋 Configuration", expanded=True):
            # Use columns to organize configuration options horizontally when possible
            col1, col2 = st.columns(2)
            
            with col1:
                # Analysis type selection
                analysis_type = st.radio(
                    "Select analysis type",
                    ["Top 100 Scholars", "Top 100 Institutions"],
                    help="Select the type of entity you want to analyze"
                )
                
                # Year selection
                start_year, end_year = st.slider(
                    "Select year range",
                    min_value=2010,
                    max_value=2025,
                    value=(2020, 2025),
                    help="Select the year range for your analysis"
                )
            
            with col2:
                # Get areas and conferences information
                areas_info = self.get_areas_and_conferences()
                
                # Research area multi-select
                selected_top_level_areas = st.multiselect(
                    "Select research areas",
                    list(areas_info.keys()),
                    help="Select research areas you're interested in"
                )
                
                # Update available conferences based on selected areas
                all_available_conferences = []
                if selected_top_level_areas:
                    for area in selected_top_level_areas:
                        all_available_conferences.extend(areas_info[area]["conferences"])
                
                selected_conferences = st.multiselect(
                    "Select conferences",
                    list(set(all_available_conferences)),
                    help="Select specific conferences you want to analyze"
                )
            
            # Execute analysis button - centered and more prominent
            _, btn_col, _ = st.columns([1, 2, 1])
            with btn_col:
                if st.button("Start Analysis", type="primary", use_container_width=True):
                    st.session_state.start_analysis = True
                    # 修复：每次点击按钮时更新会话状态中的参数
                    st.session_state.analysis_type = analysis_type
                    st.session_state.start_year = start_year
                    st.session_state.end_year = end_year
                    st.session_state.selected_conferences = selected_conferences
                    
        # Results section - using session state to maintain state between reruns
        if st.session_state.start_analysis:
            # Horizontal line to separate configuration from results
            st.markdown("---")
            
            # 移除了此处的条件检查，直接使用会话状态中最新的参数
            
            # Filter articles
            filtered_articles = self.filter_articles(
                st.session_state.selected_conferences,
                st.session_state.start_year,
                st.session_state.end_year
            )

            if not filtered_articles:
                st.warning("No articles found matching your criteria. We'll still try to display any available data.")
                # Continue execution even if no articles found
                # This allows showing partial or empty results instead of stopping

            # Choose analysis method based on type
            if st.session_state.analysis_type == "Top 100 Institutions":
                # Analyze top institutions - display all institutions with publications
                top_institutions, inst_yearly_counts, inst_top_authors = self.analyze_top_institutions(
                    filtered_articles, top_n=None)

                st.write(f"Found {len(top_institutions)} institutions with publications meeting the criteria")

                # Create dataframe for display - even if empty, we'll create a structure
                inst_data = []
                # Limit to displaying at most 100 institutions
                display_institutions = top_institutions[:100]

                for rank, (inst, total) in enumerate(display_institutions, 1):
                    row = {
                        "Rank": rank,
                        "Institution": inst,
                        "Total Papers": total
                    }

                    # Add yearly paper counts
                    for year in range(st.session_state.start_year, st.session_state.end_year + 1):
                        row[str(year)] = inst_yearly_counts[inst].get(year, 0)

                    # Add Top 10 authors
                    top_10_authors = inst_top_authors[inst]
                    row["Top 10 Authors"] = ", ".join([f"{author}({count})" for author, count in top_10_authors])

                    inst_data.append(row)

                # Even if we have no data, create an empty DataFrame with proper columns
                if not inst_data:
                    columns = ["Rank", "Institution", "Total Papers", "Top 10 Authors"]
                    columns.extend([str(year) for year in range(st.session_state.start_year, st.session_state.end_year + 1)])
                    df = pd.DataFrame(columns=columns)
                    st.info("No institutions found with papers matching your criteria.")
                else:
                    df = pd.DataFrame(inst_data)

                    # Display results
                    st.header("Institution Analysis Results")

                    # Institution paper count table
                    with st.expander(f"Institutions with Publications ({len(df)} found)", expanded=True):
                        st.dataframe(df.set_index('Rank'), use_container_width=True)

                    # Only display charts if we have data
                    if not df.empty:
                        # Institution paper count bar chart
                        fig = px.bar(
                            df,
                            x='Institution',
                            y='Total Papers',
                            title=f"{st.session_state.start_year}-{st.session_state.end_year} US Institutions with Publications in Selected Conferences",
                            labels={'Total Papers': 'Paper Count', 'Institution': 'Institution Name'}
                        )
                        st.plotly_chart(fig, use_container_width=True)

                        # Yearly trends - only top 5 institutions or all if less than 5
                        top_n = min(5, len(df))
                        if top_n > 0:
                            top5_df = df.head(top_n)
                            # Ensure data for all years
                            year_columns = [str(year) for year in range(st.session_state.start_year, st.session_state.end_year + 1)]
                            yearly_trend = top5_df[year_columns]

                            # Create yearly data for each institution
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
                                title=f"{st.session_state.start_year}-{st.session_state.end_year} Top {top_n} Institutions Yearly Paper Trends",
                                markers=True
                            )
                            # Improve line visibility and ensure x-axis shows all years
                            trend_fig.update_traces(line=dict(width=3))
                            trend_fig.update_xaxes(tickvals=year_columns)
                            st.plotly_chart(trend_fig, use_container_width=True)
                        else:
                            st.info("No trend data available for visualization.")

            else:  # Top 100 Scholars
                # Analyze top authors - display all scholars with publications
                top_authors, author_yearly_counts = self.analyze_top_authors(filtered_articles, top_n=None)

                st.write(f"Found {len(top_authors)} scholars with publications meeting the criteria")

                # Create dataframe for display - even if empty, we'll create a structure
                author_data = []
                # Limit to displaying at most 100 scholars
                display_authors = top_authors[:100]

                for rank, (author, total) in enumerate(display_authors, 1):
                    row = {
                        "Rank": rank,
                        "Author": author,
                        "Institution": self.author_institutions.get(author, "Unknown"),
                        "Total Papers": total
                    }

                    # Add yearly paper counts
                    for year in range(st.session_state.start_year, st.session_state.end_year + 1):
                        row[str(year)] = author_yearly_counts[author].get(year, 0)

                    author_data.append(row)

                # Even if we have no data, create an empty DataFrame with proper columns
                if not author_data:
                    columns = ["Rank", "Author", "Institution", "Total Papers"]
                    columns.extend([str(year) for year in range(st.session_state.start_year, st.session_state.end_year + 1)])
                    df = pd.DataFrame(columns=columns)
                    st.info("No authors found with papers matching your criteria.")
                else:
                    df = pd.DataFrame(author_data)

                    # Display results
                    st.header("Scholar Analysis Results")

                    # Author paper count table
                    with st.expander(f"Scholars with Publications ({len(df)} found)", expanded=True):
                        st.dataframe(df.set_index('Rank'), use_container_width=True)

                    # Only display charts if we have data
                    if not df.empty:
                        # Author paper count bar chart
                        fig = px.bar(
                            df,
                            x='Author',
                            y='Total Papers',
                            title=f"{st.session_state.start_year}-{st.session_state.end_year} US Authors with Publications in Selected Conferences",
                            labels={'Total Papers': 'Paper Count', 'Author': 'Author Name'}
                        )
                        st.plotly_chart(fig, use_container_width=True)

                        # Yearly trends - only top 5 authors or all if less than 5
                        top_n = min(5, len(df))
                        if top_n > 0:
                            top5_df = df.head(top_n)
                            # Ensure data for all years
                            year_columns = [str(year) for year in range(st.session_state.start_year, st.session_state.end_year + 1)]

                            # Create yearly data for each author
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
                                title=f"{st.session_state.start_year}-{st.session_state.end_year} Top {top_n} Scholars Yearly Paper Trends",
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

        # If conferences are selected, add them to the mapping
        selected_confs_with_aliases = set()
        alias_to_canonical = {}

        if selected_conferences:
            for conf in selected_conferences:
                selected_confs_with_aliases.add(conf)
                # Add all aliases for this conference
                for alias, canonical in self.conf_aliases.items():
                    if canonical == conf:
                        selected_confs_with_aliases.add(alias)
                        alias_to_canonical[alias] = canonical

        for article in self.articles:
            conf_orig = article.get("conf")
            year = int(article.get("year", 0))

            # Apply conference alias conversion
            conf = conf_orig
            if conf in self.conf_aliases:
                conf = self.conf_aliases[conf]
                # Preserve original name before processing
                article["original_conf"] = conf_orig

            # If no conferences selected or the conference is in the selected list, and year is in range
            if (not selected_conferences or conf in selected_confs_with_aliases) and \
                    start_year <= year <= end_year:
                filtered_articles.append(article)

                # Use normalized conference name for counting
                conf_for_counting = conf
                if conf_orig in alias_to_canonical:
                    conf_for_counting = alias_to_canonical[conf_orig]

                conference_counts[conf_for_counting] += 1

        # Print number of matching articles and conference distribution
        st.write(f"Found {len(filtered_articles)} articles matching the criteria")
        if filtered_articles:
            st.write("Conference distribution:")
            # Sort conferences by count
            sorted_confs = sorted(conference_counts.items(), key=lambda x: x[1], reverse=True)
            for conf, count in sorted_confs:
                st.write(f"  - {conf}: {count} papers")

            # Show alias usage
            alias_usage = defaultdict(int)
            for article in filtered_articles:
                if "original_conf" in article and article["original_conf"] != article.get("conf"):
                    alias_usage[f"{article['original_conf']} -> {article.get('conf')}"] += 1

            if alias_usage:
                st.write("Conference alias usage:")
                for alias_map, count in alias_usage.items():
                    st.write(f"  - {alias_map}: {count} papers")

        return filtered_articles

    def analyze_top_authors(self, filtered_articles, top_n=100):
        """Analyze top authors"""
        # If top_n is None, display all authors with publications
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

        # If top_n is None, show all results; otherwise, take the top_n
        top_authors = sorted_authors if top_n is None else sorted_authors[:top_n]

        return top_authors, author_yearly_counts

    def analyze_top_institutions(self, filtered_articles, top_n=50):
        """Analyze top institutions"""
        # If top_n is None, display all institutions with publications
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

        # If top_n is None, show all results; otherwise, take the top_n
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
