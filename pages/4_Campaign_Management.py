
import streamlit as st
import pandas as pd
import sys
from pathlib import Path
from datetime import date

# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Campaign Management | ChurnIQ",
    page_icon="📢",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parents[1]

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.campaign_manager import (
    initialize_database,
    add_campaign,
    get_all_campaigns,
    update_response,
    update_outcome,
    delete_campaign
)

initialize_database()

# =========================================================
# 2. CUSTOM STYLING
# =========================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background-color: #F5F7FB;
}

section[data-testid="stSidebar"] {
    background-color: #111827;
}

section[data-testid="stSidebar"] * {
    color: #F9FAFB;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1450px;
}

h1, h2, h3 {
    color: #111827;
    font-weight: 800;
}

p, label {
    color: #64748B;
}

div[data-testid="stMetric"] {
    background: white;
    border: 1px solid #E5EAF2;
    padding: 20px;
    border-radius: 14px;
    box-shadow: 0 2px 8px rgba(15, 23, 42, 0.03);
}

div[data-testid="stMetricLabel"] {
    color: #64748B;
    font-weight: 600;
}

div[data-testid="stMetricValue"] {
    color: #111827;
    font-weight: 800;
}

div[data-testid="stForm"] {
    background: white;
    border: 1px solid #E5EAF2;
    padding: 24px;
    border-radius: 14px;
}

div.stButton > button,
div.stDownloadButton > button,
div[data-testid="stFormSubmitButton"] > button {
    border-radius: 9px;
    font-weight: 700;
    min-height: 42px;
}

div[data-testid="stDataFrame"] {
    border: 1px solid #E5EAF2;
    border-radius: 12px;
    overflow: hidden;
}

hr {
    border-color: #E5EAF2;
}
</style>
""", unsafe_allow_html=True)


# =========================================================
# 3. HELPER FUNCTIONS
# =========================================================

def section_heading(title, subtitle=None):
    st.markdown(f"### {title}")
    if subtitle:
        st.caption(subtitle)


def load_campaigns():
    data = get_all_campaigns()

    if data is None:
        return pd.DataFrame()

    return data


def get_count(data, column, value):
    if data.empty or column not in data.columns:
        return 0

    return int((data[column] == value).sum())

st.markdown(
    """
    <div style="background-color:#111827; padding:30px; border-radius:18px;">
        <h1 style="color:white;">Campaign Management</h1>
        <p style="color:#CBD5E1;">
            Create retention campaigns, track customer responses
            and measure campaign outcomes.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# =========================================================
# 5. CAMPAIGN OVERVIEW
# =========================================================

campaigns = load_campaigns()

total_campaigns = len(campaigns)
accepted = get_count(campaigns, "response", "Accepted")
pending = get_count(campaigns, "response", "Pending")
retained = get_count(campaigns, "outcome", "Retained")

section_heading(
    "Campaign Overview",
    "A quick snapshot of your customer retention activities."
)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Campaigns", total_campaigns)

with col2:
    st.metric(
        "Accepted Offers",
        accepted,
        delta=f"{accepted / total_campaigns * 100:.1f}% of campaigns"
        if total_campaigns else None
    )

with col3:
    st.metric("Pending Responses", pending)

with col4:
    st.metric("Customers Retained", retained)

st.divider()


# =========================================================
# 6. CREATE NEW CAMPAIGN
# =========================================================

section_heading(
    "Create New Campaign",
    "Launch a retention offer for an at-risk customer."
)

with st.form("campaign_form", clear_on_submit=True):

    col1, col2 = st.columns(2)

    with col1:
        customer_id = st.text_input(
            "Customer ID",
            placeholder="Example: 7590-VHVEG"
        )

        campaign_name = st.selectbox(
            "Campaign Type",
            [
                "Loyalty Discount",
                "Annual Contract Offer",
                "Technical Support Offer",
                "Online Security Offer",
                "Special Retention Offer",
                "Customer Satisfaction Campaign"
            ]
        )

    with col2:
        campaign_date = st.date_input(
            "Campaign Date",
            value=date.today()
        )

        offer_details = st.text_area(
            "Offer Details",
            placeholder="Describe the offer provided to the customer...",
            height=100
        )

    submitted = st.form_submit_button(
        "＋ Create Campaign",
        use_container_width=True
    )

    if submitted:

        if not customer_id.strip():
            st.error("Please enter a Customer ID.")

        elif not offer_details.strip():
            st.error("Please enter offer details.")

        else:
            add_campaign(
                customer_id.strip(),
                campaign_name,
                offer_details.strip(),
                campaign_date.strftime("%Y-%m-%d")
            )

            st.success("Campaign created successfully!")
            st.rerun()

st.divider()


# =========================================================
# 7. CAMPAIGN RECORDS
# =========================================================

campaigns = load_campaigns()

section_heading(
    "Campaign Records",
    "Search and review all customer retention campaigns."
)

if campaigns.empty:

    st.info("No campaigns available. Create your first campaign above.")

else:

    # Search and filter
    filter_col1, filter_col2, filter_col3 = st.columns(3)

    with filter_col1:
        search_customer = st.text_input(
            "Search Customer ID",
            placeholder="Enter customer ID..."
        )

    with filter_col2:
        response_filter = st.selectbox(
            "Filter by Response",
            ["All", "Pending", "Accepted", "Rejected"]
        )

    with filter_col3:
        outcome_filter = st.selectbox(
            "Filter by Outcome",
            ["All", "Not Evaluated", "Retained", "Churned"]
        )

    filtered = campaigns.copy()

    if search_customer:
        filtered = filtered[
            filtered["customer_id"].astype(str)
            .str.contains(search_customer, case=False, na=False)
        ]

    if response_filter != "All":
        filtered = filtered[
            filtered["response"] == response_filter
        ]

    if outcome_filter != "All":
        filtered = filtered[
            filtered["outcome"] == outcome_filter
        ]

    display_columns = [
        "id",
        "customer_id",
        "campaign_name",
        "offer_details",
        "campaign_date",
        "response",
        "outcome"
    ]

    available_columns = [
        column for column in display_columns
        if column in filtered.columns
    ]

    st.caption(f"Showing {len(filtered)} of {len(campaigns)} campaigns")

    st.dataframe(
        filtered[available_columns],
        use_container_width=True,
        hide_index=True
    )

    st.divider()


    # =====================================================
    # 8. UPDATE CAMPAIGN RESPONSE
    # =====================================================

    section_heading(
        "Update Customer Response",
        "Record whether the customer accepted or rejected the offer."
    )

    campaign_ids = campaigns["id"].tolist()

    selected_id = st.selectbox(
        "Select Campaign",
        campaign_ids,
        format_func=lambda x: (
            f"Campaign #{x} — "
            f"{campaigns.loc[campaigns['id'] == x, 'customer_id'].iloc[0]}"
        ),
        key="response_campaign_id"
    )

    selected_campaign = campaigns[
        campaigns["id"] == selected_id
    ].iloc[0]

    st.info(
        f"Selected customer: **{selected_campaign['customer_id']}**  |  "
        f"Campaign: **{selected_campaign['campaign_name']}**"
    )

    response_options = ["Pending", "Accepted", "Rejected"]

    current_response = selected_campaign["response"]

    new_response = st.selectbox(
        "Customer Response",
        response_options,
        index=(
            response_options.index(current_response)
            if current_response in response_options else 0
        ),
        key="new_response"
    )

    if st.button("Update Response", key="update_response_button"):

        update_response(selected_id, new_response)

        st.success("Customer response updated!")
        st.rerun()

    st.divider()


    # =====================================================
    # 9. UPDATE RETENTION OUTCOME
    # =====================================================

    section_heading(
        "Update Retention Outcome",
        "Record the final result of your retention campaign."
    )

    outcome_id = st.selectbox(
        "Select Campaign",
        campaign_ids,
        format_func=lambda x: (
            f"Campaign #{x} — "
            f"{campaigns.loc[campaigns['id'] == x, 'customer_id'].iloc[0]}"
        ),
        key="outcome_campaign_id"
    )

    selected_outcome = campaigns[
        campaigns["id"] == outcome_id
    ].iloc[0]

    st.info(
        f"Selected customer: **{selected_outcome['customer_id']}**  |  "
        f"Current response: **{selected_outcome['response']}**"
    )

    outcome_options = ["Not Evaluated", "Retained", "Churned"]

    current_outcome = selected_outcome["outcome"]

    new_outcome = st.selectbox(
        "Retention Outcome",
        outcome_options,
        index=(
            outcome_options.index(current_outcome)
            if current_outcome in outcome_options else 0
        ),
        key="new_outcome"
    )

    if st.button("Update Outcome", key="update_outcome_button"):

        update_outcome(outcome_id, new_outcome)

        st.success("Retention outcome updated!")
        st.rerun()

    st.divider()


    # =====================================================
    # 10. DELETE CAMPAIGN
    # =====================================================

    section_heading(
        "Delete Campaign",
        "Remove an unwanted campaign record."
    )

    with st.expander("Open campaign deletion controls"):

        delete_id = st.selectbox(
            "Select Campaign to Delete",
            campaign_ids,
            format_func=lambda x: (
                f"Campaign #{x} — "
                f"{campaigns.loc[campaigns['id'] == x, 'customer_id'].iloc[0]}"
            ),
            key="delete_campaign_id"
        )

        confirm_delete = st.checkbox(
            "I understand this action cannot be undone.",
            key="confirm_delete"
        )

        if st.button(
            "Delete Selected Campaign",
            type="primary",
            key="delete_campaign_button"
        ):

            if confirm_delete:
                delete_campaign(delete_id)
                st.success("Campaign deleted successfully!")
                st.rerun()

            else:
                st.warning("Please confirm before deleting.")

    st.divider()


    # =====================================================
    # 11. CAMPAIGN ANALYTICS
    # =====================================================

    campaigns = load_campaigns()

    section_heading(
        "Campaign Performance",
        "Understand customer responses and retention outcomes."
    )

    if campaigns.empty:
        st.info("No campaign data available for analytics.")

    else:

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Customer Response Distribution**")

            response_counts = (
                campaigns["response"]
                .value_counts()
                .rename_axis("Response")
                .reset_index(name="Count")
            )

            st.bar_chart(
                response_counts,
                x="Response",
                y="Count",
                color="#3B82F6"
            )

        with col2:
            st.markdown("**Retention Outcome Distribution**")

            outcome_counts = (
                campaigns["outcome"]
                .value_counts()
                .rename_axis("Outcome")
                .reset_index(name="Count")
            )

            st.bar_chart(
                outcome_counts,
                x="Outcome",
                y="Count",
                color="#10B981"
            )

        # Retention rate
        evaluated = campaigns[
            campaigns["outcome"].isin(["Retained", "Churned"])
        ]

        if not evaluated.empty:
            retention_rate = (
                (evaluated["outcome"] == "Retained").sum()
                / len(evaluated)
            ) * 100

            st.markdown("**Retention Success Rate**")
            st.progress(float(retention_rate / 100))
            st.caption(
                f"{retention_rate:.1f}% of evaluated campaign customers were retained."
            )

    # =====================================================
    # 12. EXPORT CAMPAIGN REPORT
    # =====================================================

    st.divider()

    section_heading(
        "Export Campaign Report",
        "Download your campaign records for further analysis."
    )

    csv = campaigns.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="↓ Download Campaign Report (CSV)",
        data=csv,
        file_name="campaign_report.csv",
        mime="text/csv",
        use_container_width=True
    )


# =========================================================
# 13. FOOTER
# =========================================================

st.divider()

st.caption(
    "ChurnIQ | Campaign records are stored locally using SQLite."
)
