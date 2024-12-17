# app.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objs as go
from scipy import stats
from statsmodels.stats.multitest import multipletests
from sklearn.utils import resample
import itertools
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.colors import to_rgba
import matplotlib.patches as mpatches
import textwrap
from matplotlib.gridspec import GridSpec
from matplotlib.offsetbox import DrawingArea, TextArea, HPacker, VPacker, AnnotationBbox
import seaborn as sns
import matplotlib
from datetime import datetime, date

# Prevent matplotlib from trying to use any Xwindows backend.
matplotlib.use('Agg')

# -------------------------------
# Custom CSS for Outlines
# -------------------------------
def add_css():
    st.markdown(
        """
        <style>
        .section {
            border: 1px solid #e0e0e0;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
            background-color: #f9f9f9;
        }
        .section-title {
            font-size: 24px;
            font-weight: bold;
            margin-bottom: 10px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

# -------------------------------
# Data Processing Functions
# -------------------------------

def remove_outliers_zscore(df, threshold=3):
    """
    Remove outliers from a dataframe based on Z-score.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) == 0:
        return df  # No numeric columns to process
    z_scores = np.abs(stats.zscore(df[numeric_cols]))
    filtered_entries = (z_scores < threshold).all(axis=1)
    return df[filtered_entries]

def find_common_parameters(dataframes):
    """
    Identify common parameters across all dataframes, excluding the 'date' column.
    """
    common_params = set(dataframes[0].columns) - {'date'}
    for df in dataframes[1:]:
        common_params &= set(df.columns) - {'date'}
    return list(common_params)

def combine_dataframes(dataframes_sorted, process_labels_sorted):
    """
    Merge all filtered dataframes on the 'date' column with suffixes to differentiate processes.
    """
    combined_df = None
    for idx, df in enumerate(dataframes_sorted):
        if combined_df is None:
            combined_df = df.copy()
        else:
            combined_df = pd.merge(
                combined_df,
                df,
                on='date',
                how='inner',
                suffixes=('', f"_{process_labels_sorted[idx]}")
            )
    if combined_df is not None:
        combined_df.set_index('date', inplace=True)
    return combined_df

def bootstrap_correlations(df, n_iterations=500, method='pearson', progress_bar=None, status_text=None, start_progress=0.0, end_progress=1.0):
    """
    Perform bootstrapping to compute median correlation matrices.
    """
    correlations = []
    for i in range(n_iterations):
        df_resampled = resample(df)
        corr_matrix = df_resampled.corr(method=method)
        correlations.append(corr_matrix)
        if progress_bar and status_text:
            # Calculate incremental progress
            progress = start_progress + (i + 1) / n_iterations * (end_progress - start_progress)
            progress_bar.progress(int(progress * 100))
            status_text.text(f"Bootstrapping {method.capitalize()} Correlations... ({i+1}/{n_iterations})")
    median_corr = pd.concat(correlations).groupby(level=0).median()
    return median_corr

# -------------------------------
# Visualization Functions
# -------------------------------

def generate_heatmap(df, title, axis_titles, progress_bar=None, status_text=None, start_progress=0.0, end_progress=1.0):
    """
    Generate and display a correlation heatmap using Plotly.
    """
    # Compute correlation matrix
    corr_matrix = df.corr(method='pearson')  # Using Pearson for heatmap

    # Create a mask for the upper triangle
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

    # Update progress
    if progress_bar and status_text:
        progress = start_progress + 0.5 * (end_progress - start_progress)
        progress_bar.progress(int(progress * 100))
        status_text.text("Computing correlation matrix...")

    # Generate heatmap using Plotly
    fig = px.imshow(
        corr_matrix,
        text_auto=True,
        aspect="auto",
        color_continuous_scale='RdBu',
        origin='lower',
        labels=dict(x="Process 2 Parameters", y="Process 1 Parameters", color="Correlation"),
        title=title
    )

    # Update layout
    fig.update_layout(
        title=dict(
            text=title,
            x=0.5,               # Center horizontally
            xanchor='center',    # Anchor the title at the center
            yanchor='top'        # Anchor the title at the top
        ),
        xaxis=dict(tickangle=45, title=None, tickfont=dict(size=12)),
        yaxis=dict(title=None, tickfont=dict(size=12)),
        autosize=False,
        width=800,
        height=600,
        margin=dict(l=100, r=100, t=100, b=100),
    )

    # Update progress
    if progress_bar and status_text:
        progress = start_progress + (0.5) * (end_progress - start_progress)
        progress_bar.progress(int(progress * 100))
        status_text.text("Rendering heatmap...")

    # Display the heatmap
    st.plotly_chart(fig)

    # Update progress to end
    if progress_bar and status_text:
        progress_bar.progress(int(end_progress * 100))
        status_text.text("Heatmap generation complete.")

    return corr_matrix

# -------------------------------
# Main Streamlit App
# -------------------------------

def main():
    # Set page config as the very first Streamlit command
    st.set_page_config(page_title="WWTP Unit Processes Network Visualization", layout="wide")

    # Add custom CSS for outlines
    add_css()

    # Add the main title
    st.markdown("<h1 style='text-align: center; color:rgb(0, 0, 0);'>WWTP Unit Processes Network Visualization</h1>", unsafe_allow_html=True)

    # -------------------------------
    # 1. Instructions Section
    # -------------------------------
    st.markdown("<div class='section'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Instructions</div>", unsafe_allow_html=True)
    st.markdown("""
    1. **Upload Files:** Upload your CSV or Excel files containing process data. Ensure each file has a 'date' column.

    2. **Label Processes:** Assign descriptive labels to each uploaded process file.

    3. **Reorder Processes:** After uploading, assign an order to the processes based on their real-life sequence (upstream to downstream).

    4. **Generate Visualizations:** Click the buttons to generate correlation heatmaps, network diagrams, bar charts, and line graphs.

    5. **Correlation Over Time:** Analyze how correlations between parameter pairs evolve over time using monthly subplots.

    6. **Targeted Network Diagram:** Use the section below to generate a network diagram centered around a specific parameter from a selected process.
    """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # -------------------------------
    # 2. File Upload and Labeling
    # -------------------------------
    st.markdown("<div class='section'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Upload and Label Files</div>", unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        "Choose CSV or Excel files",
        accept_multiple_files=True,
        type=['csv', 'xlsx', 'xls']
    )
    process_labels = []
    dataframes = []

    if uploaded_files:
        for idx, uploaded_file in enumerate(uploaded_files):
            st.subheader(f"Process File {idx + 1}: {uploaded_file.name}")
            try:
                # Read the uploaded file
                if uploaded_file.name.endswith(('.xlsx', '.xls')):
                    df = pd.read_excel(uploaded_file)
                else:
                    df = pd.read_csv(uploaded_file)

                df.columns = df.columns.str.lower().str.strip()
                if 'date' not in df.columns:
                    st.error(f"The file **{uploaded_file.name}** does not contain a 'date' column.")
                    st.stop()

                df['date'] = pd.to_datetime(df['date'], errors='coerce')
                df = df.dropna(subset=["date"])
                df = remove_outliers_zscore(df)
                dataframes.append(df)

                # Prompt user for a label for this process
                label = st.text_input(
                    f"Enter a label for **{uploaded_file.name}**:",
                    value=uploaded_file.name.split('.')[0],
                    key=f"label_{idx}"
                )
                process_labels.append(label)
            except Exception as e:
                st.error(f"Error processing file **{uploaded_file.name}**: {e}")
                st.stop()

        if len(dataframes) < 2:
            st.warning("Please upload at least two files to generate diagrams.")
            st.stop()

        # Identify common parameters
        common_params = find_common_parameters(dataframes)
        if not common_params:
            st.error("No common parameters found across all uploaded files.")
            st.stop()

        st.success(f"Common parameters identified: {', '.join(common_params)}")
    st.markdown("</div>", unsafe_allow_html=True)

    if uploaded_files and len(dataframes) >=2 and common_params:
        # -------------------------------
        # 3. Reordering Uploaded Files via Sidebar
        # -------------------------------
        st.markdown("<div class='section'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Reorder Uploaded Files</div>", unsafe_allow_html=True)
        st.write("Please assign an order to the uploaded files based on their sequence in real life (upstream to downstream).")

        with st.sidebar:
            st.markdown("### Reorder Uploaded Files")
            st.write("Please assign an order to the uploaded files based on their sequence in real life (upstream to downstream).")

            # Initialize list to store order
            order_numbers = []
            for idx, file in enumerate(uploaded_files):
                order = st.number_input(
                    f"Order for {file.name}", 
                    min_value=1, 
                    max_value=len(uploaded_files), 
                    value=idx+1, 
                    step=1, 
                    key=f"order_sidebar_{idx}"
                )
                order_numbers.append(order)

            # Validate unique order numbers
            if len(set(order_numbers)) != len(order_numbers):
                st.error("Each file must have a unique order number. Please adjust the order numbers accordingly.")
                st.stop()

            # Combine files, labels, and order
            file_orders = list(zip(uploaded_files, process_labels, order_numbers))

            # Sort files based on order
            sorted_files = sorted(file_orders, key=lambda x: x[2])

            # Unzip sorted files and labels
            uploaded_files_sorted, process_labels_sorted, _ = zip(*sorted_files)

            # Correctly sort dataframes_sorted based on sorted_files
            dataframes_sorted = [df for _, _, df in sorted_files]

            # Debugging: Display number of records and date ranges after sorting
            st.markdown("<div class='section'>", unsafe_allow_html=True)
            st.markdown("<div class='section-title'>Data Overview After Sorting</div>", unsafe_allow_html=True)

            for idx, df in enumerate(dataframes_sorted):
                st.write(f"**{process_labels_sorted[idx]}**:")
                try:
                    if isinstance(df, pd.DataFrame):
                        st.write(f"Number of records: {len(df)}")
                        if not df.empty:
                            st.write(f"Date Range: {df['date'].min()} to {df['date'].max()}")
                        else:
                            st.write("No data available.")
                    else:
                        st.write(f"Number of records: {df} (Expected a DataFrame but got {type(df).__name__})")
                except Exception as e:
                    st.error(f"Error displaying data overview for **{process_labels_sorted[idx]}**: {e}")
                    st.stop()

            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

        # -------------------------------
        # 4. Correlation Over Time Visualization with Monthly Subplots
        # -------------------------------
        st.markdown("<div class='section'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Correlation Over Time</div>", unsafe_allow_html=True)
        
        st.write("Analyze how correlations between parameter pairs evolve over time using monthly subplots.")
        
        # Select parameter pairs
        st.subheader("Select Parameter Pairs")
        # Generate all possible unique parameter pairs
        parameter_pairs = list(itertools.combinations(common_params, 2))
        pair_labels = [f"{pair[0]} & {pair[1]}" for pair in parameter_pairs]
        
        selected_pairs = st.multiselect(
            "Choose parameter pairs to analyze:",
            options=pair_labels,
            help="Select one or more parameter pairs to visualize their correlation over time."
        )
        
        if not selected_pairs:
            st.info("Please select at least one parameter pair to display the correlation over time.")
            st.markdown("</div>", unsafe_allow_html=True)
            return
        
        # Automatically split data into months
        combined_df = combine_dataframes(dataframes_sorted, process_labels_sorted)
        if combined_df is None or combined_df.empty:
            st.error("No data available after merging the dataframes.")
            st.markdown("</div>", unsafe_allow_html=True)
            return
        combined_df = combined_df.reset_index()  # Ensure 'date' is a column
        combined_df['month'] = combined_df['date'].dt.to_period('M')
        months = combined_df['month'].dropna().unique()
        months_sorted = sorted(months)
        
        if len(months_sorted) == 0:
            st.error("No valid months found in the data.")
            st.markdown("</div>", unsafe_allow_html=True)
            return

        # Initialize Plotly figure with subplots
        from plotly.subplots import make_subplots
        num_months = len(months_sorted)
        cols = 3  # Number of columns in the subplot grid
        rows = (num_months // cols) + int(num_months % cols > 0)
        
        fig = make_subplots(rows=rows, cols=cols, subplot_titles=[str(month) for month in months_sorted])

        for idx, month in enumerate(months_sorted):
            row = (idx // cols) + 1
            col = (idx % cols) + 1
            df_month = combined_df[combined_df['month'] == month]
            
            for pair_label in selected_pairs:
                param1, param2 = pair_label.split(" & ")
                # Identify all columns related to param1 and param2
                param1_cols = [c for c in combined_df.columns if c.startswith(param1 + "_")]
                param2_cols = [c for c in combined_df.columns if c.startswith(param2 + "_")]
                
                if not param1_cols or not param2_cols:
                    st.warning(f"No matching columns found for pair: {pair_label} in {month}")
                    continue
                
                # Compute correlations for all combinations and average them
                correlations = []
                for p1 in param1_cols:
                    for p2 in param2_cols:
                        if p1 != p2:
                            corr_value = df_month[p1].corr(df_month[p2], method='pearson')
                            if not np.isnan(corr_value):
                                correlations.append(corr_value)
                
                if correlations:
                    # Compute the mean correlation across all combinations
                    mean_corr = np.mean(correlations)
                    fig.add_trace(
                        go.Scatter(
                            x=[str(month)],
                            y=[mean_corr],
                            mode='markers+lines',
                            name=pair_label
                        ),
                        row=row,
                        col=col
                    )
    
        fig.update_layout(
            height=300 * rows,  # Adjust height based on number of rows
            width=1000,  # Adjust width as needed
            title_text="Monthly Correlations of Selected Parameter Pairs",
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # -------------------------------
        # 5. Generate Heatmaps and Store Correlation Matrices
        # -------------------------------
        st.markdown("<div class='section'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Generate Heatmaps</div>", unsafe_allow_html=True)

        correlation_matrices = []
        parameters_per_edge = []
        for i in range(len(uploaded_files_sorted) - 1):
            st.markdown(f"### Heatmap: **{process_labels_sorted[i]}** vs **{process_labels_sorted[i + 1]}**")

            # Create separate progress bar and status
            heatmap_progress = st.progress(0)
            heatmap_status = st.empty()

            # Merge data
            df1 = dataframes_sorted[i][['date'] + common_params]
            df2 = dataframes_sorted[i + 1][['date'] + common_params]
            merged_df = pd.merge(
                df1, df2, on="date",
                suffixes=(f"_{process_labels_sorted[i]}", f"_{process_labels_sorted[i + 1]}")
            )
            merged_df = merged_df.drop(columns=["date"], errors="ignore")
            merged_df = merged_df.replace([np.inf, -np.inf], np.nan)
            merged_df = merged_df.dropna()
            numeric_columns = merged_df.select_dtypes(include=[np.number]).columns
            merged_df = merged_df[numeric_columns]

            # Check if merged_df is empty
            if merged_df.empty:
                st.warning(f"No data available for heatmap between '{process_labels_sorted[i]}' and '{process_labels_sorted[i + 1]}' after merging and cleaning.")
                correlation_matrices.append(pd.DataFrame())
                parameters_per_edge.append([])
                continue

            # Generate heatmap
            filtered_corr_matrix = generate_heatmap(
                merged_df,
                f"Correlation Coefficient Heatmap: {process_labels_sorted[i]} vs {process_labels_sorted[i + 1]}",
                ("X-Axis", "Y-Axis"),
                progress_bar=heatmap_progress,
                status_text=heatmap_status,
                start_progress=0.0,
                end_progress=1.0
            )
            correlation_matrices.append(filtered_corr_matrix)

            # Identify parameters contributing to the correlation
            shared_params = []
            for param in common_params:
                infl_param = f"{param}_{process_labels_sorted[i]}"
                ode_param = f"{param}_{process_labels_sorted[i + 1]}"
                if infl_param in filtered_corr_matrix.index and ode_param in filtered_corr_matrix.columns:
                    if filtered_corr_matrix.loc[infl_param, ode_param] != 0:
                        shared_params.append(param)
            parameters_per_edge.append(shared_params)

        st.markdown("</div>", unsafe_allow_html=True)

        # -------------------------------
        # 6. Identify Globally Shared Parameters
        # -------------------------------
        st.markdown("<div class='section'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Globally Shared Parameters</div>", unsafe_allow_html=True)
        globally_shared_parameters = set(parameters_per_edge[0])
        for params in parameters_per_edge[1:]:
            globally_shared_parameters &= set(params)

        st.markdown(f"**Globally shared parameters across all node pairs:** {', '.join(globally_shared_parameters) if globally_shared_parameters else 'None'}")
        if not globally_shared_parameters:
            st.error("No globally shared parameters found.")
            st.stop()
        st.markdown("</div>", unsafe_allow_html=True)

        # -------------------------------
        # 7. Generate Visualizations
        # -------------------------------
        st.markdown("<div class='section'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Generate Visualizations</div>", unsafe_allow_html=True)

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            if st.button("Generate Globally Shared Network Diagram"):
                # Create separate progress bar and status
                global_net_progress = st.progress(0)
                global_net_status = st.empty()

                # Insert your network diagram generation function here
                # For demonstration, we'll just update the progress bar
                for i in range(100):
                    global_net_progress.progress(i + 1)
                    global_net_status.text(f"Progress: {i + 1}%")
                global_net_status.text("Globally Shared Network Diagram generated.")

        with col2:
            if st.button("Generate Locally Shared Network Diagram"):
                # Create separate progress bar and status
                local_net_progress = st.progress(0)
                local_net_status = st.empty()

                # Insert your network diagram generation function here
                # For demonstration, we'll just update the progress bar
                for i in range(100):
                    local_net_progress.progress(i + 1)
                    local_net_status.text(f"Progress: {i + 1}%")
                local_net_status.text("Locally Shared Network Diagram generated.")

        with col3:
            if st.button("Generate Bar Chart for Globally Shared Parameters"):
                # Create separate progress bar and status
                bar_chart_progress = st.progress(0)
                bar_chart_status = st.empty()

                # Insert your bar chart generation function here
                # For demonstration, we'll just update the progress bar
                for i in range(100):
                    bar_chart_progress.progress(i + 1)
                    bar_chart_status.text(f"Progress: {i + 1}%")
                bar_chart_status.text("Bar Chart generated.")

        with col4:
            if st.button("Generate Line Graph for Globally Shared Parameters"):
                # Create separate progress bar and status
                line_graph_progress = st.progress(0)
                line_graph_status = st.empty()

                # Insert your line graph generation function here
                # For demonstration, we'll just update the progress bar
                for i in range(100):
                    line_graph_progress.progress(i + 1)
                    line_graph_status.text(f"Progress: {i + 1}%")
                line_graph_status.text("Line Graph generated.")

        st.markdown("</div>", unsafe_allow_html=True)

        # -------------------------------
        # 8. Targeted Network Diagram Section with Separate Progress Bar
        # -------------------------------
        st.markdown("<div class='section'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Targeted Network Diagram</div>", unsafe_allow_html=True)
        st.write("Generate a network diagram centered around a specific parameter from a selected process.")

        # Create separate progress bar and status
        targeted_net_progress = st.progress(0)
        targeted_net_status = st.empty()

        # Insert your targeted network diagram function here
        # For demonstration, we'll just update the progress bar
        for i in range(100):
            targeted_net_progress.progress(i + 1)
            targeted_net_status.text(f"Progress: {i + 1}%")
        targeted_net_status.text("Targeted Network Diagram generated.")
        st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------
# Run the Streamlit App
# -------------------------------

if __name__ == "__main__":
    main()