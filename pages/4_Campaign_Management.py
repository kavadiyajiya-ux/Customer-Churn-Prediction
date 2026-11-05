import streamlit as st
import pandas as pd
import sys
from pathlib import Path
from datetime import date

# ----------------------------------
# 1. PAGE CONFIGURATION
# ----------------------------------

st.set_page_config(
    page_title="Campaign Management",
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

# ----------------------------------
# 2. PAGE TITLE
# ----------------------------------

st.title("📢 Campaign Management")

st.caption(
    "Create, monitor and analyze customer retention campaigns."
)

st.divider()

# ----------------------------------
# 3. CAMPAIGN OVERVIEW
# ----------------------------------

campaigns = get_all_campaigns()

total_campaigns = len(campaigns)

accepted = (
    (campaigns["response"] == "Accepted").sum()
    if not campaigns.empty else 0
)

pending = (
    (campaigns["response"] == "Pending").sum()
    if not campaigns.empty else 0
)

retained = (
    (campaigns["outcome"] == "Retained").sum()
    if not campaigns.empty else 0
)

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Campaigns", total_campaigns)
col2.metric("Accepted Offers", accepted)
col3.metric("Pending Responses", pending)
col4.metric("Customers Retained", retained)

st.divider()

# ----------------------------------
# 4. CREATE NEW CAMPAIGN
# ----------------------------------

st.subheader("➕ Create New Campaign")

with st.form("campaign_form"):

    customer_id = st.text_input(
        "Customer ID",
        placeholder="Example: 7590-VHVEG"
    )

    campaign_name = st.selectbox(
        "Campaign Name",
        [
            "Loyalty Discount",
            "Annual Contract Offer",
            "Technical Support Offer",
            "Online Security Offer",
            "Special Retention Offer",
            "Customer Satisfaction Campaign"
        ]
    )

    offer_details = st.text_area(
        "Offer Details",
        placeholder="Describe the offer provided to the customer..."
    )

    campaign_date = st.date_input(
        "Campaign Date",
        value=date.today()
    )

    submitted = st.form_submit_button(
        "Create Campaign",
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

# ----------------------------------
# 5. VIEW ALL CAMPAIGNS
# ----------------------------------

st.subheader("📋 Campaign Records")

campaigns = get_all_campaigns()

if campaigns.empty:

    st.info(
        "No campaigns available. Create your first campaign above."
    )

else:

    display_columns = [
        "id",
        "customer_id",
        "campaign_name",
        "offer_details",
        "campaign_date",
        "response",
        "outcome"
    ]

    st.dataframe(
        campaigns[display_columns],
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    # ----------------------------------
    # 6. UPDATE CAMPAIGN RESPONSE
    # ----------------------------------

    st.subheader("✏️ Update Campaign Response")

    campaign_ids = campaigns["id"].tolist()

    selected_id = st.selectbox(
        "Select Campaign ID",
        campaign_ids,
        key="response_campaign_id"
    )

    selected_campaign = campaigns[
        campaigns["id"] == selected_id
    ].iloc[0]

    st.write(
        f"Customer: **{selected_campaign['customer_id']}**"
    )

    response_options = [
        "Pending",
        "Accepted",
        "Rejected"
    ]

    current_response = selected_campaign["response"]

    new_response = st.selectbox(
        "Customer Response",
        response_options,
        index=(
            response_options.index(current_response)
            if current_response in response_options else 0
        )
    )

    if st.button("Update Response"):

        update_response(
            selected_id,
            new_response
        )

        st.success("Customer response updated!")
        st.rerun()

    st.divider()

    # ----------------------------------
    # 7. UPDATE RETENTION OUTCOME
    # ----------------------------------

    st.subheader("🎯 Update Retention Outcome")

    outcome_id = st.selectbox(
        "Select Campaign",
        campaign_ids,
        key="outcome_campaign_id"
    )

    selected_outcome = campaigns[
        campaigns["id"] == outcome_id
    ].iloc[0]

    st.write(
        f"Customer: **{selected_outcome['customer_id']}**"
    )

    outcome_options = [
        "Not Evaluated",
        "Retained",
        "Churned"
    ]

    current_outcome = selected_outcome["outcome"]

    new_outcome = st.selectbox(
        "Retention Outcome",
        outcome_options,
        index=(
            outcome_options.index(current_outcome)
            if current_outcome in outcome_options else 0
        )
    )

    if st.button("Update Outcome"):

        update_outcome(
            outcome_id,
            new_outcome
        )

        st.success("Retention outcome updated!")
        st.rerun()

    st.divider()

    # ----------------------------------
    # 8. DELETE CAMPAIGN
    # ----------------------------------

    st.subheader("🗑️ Delete Campaign")

    delete_id = st.selectbox(
        "Select Campaign to Delete",
        campaign_ids,
        key="delete_campaign_id"
    )

    confirm_delete = st.checkbox(
        "I confirm that I want to delete this campaign."
    )

    if st.button("Delete Selected Campaign"):

        if confirm_delete:

            delete_campaign(delete_id)

            st.success("Campaign deleted successfully!")
            st.rerun()

        else:
            st.warning("Please confirm before deleting.")

    st.divider()

    # ----------------------------------
    # 9. CAMPAIGN ANALYTICS
    # ----------------------------------

    st.subheader("📊 Campaign Performance")

    campaigns = get_all_campaigns()

    col1, col2 = st.columns(2)

    with col1:

        st.write("**Customer Response Distribution**")

        response_counts = (
            campaigns["response"]
            .value_counts()
            .reset_index()
        )

        response_counts.columns = [
            "Response",
            "Count"
        ]

        st.bar_chart(
            response_counts,
            x="Response",
            y="Count"
        )

    with col2:

        st.write("**Retention Outcome Distribution**")

        outcome_counts = (
            campaigns["outcome"]
            .value_counts()
            .reset_index()
        )

        outcome_counts.columns = [
            "Outcome",
            "Count"
        ]

        st.bar_chart(
            outcome_counts,
            x="Outcome",
            y="Count"
        )

    # ----------------------------------
    # 10. DOWNLOAD CAMPAIGN REPORT
    # ----------------------------------

    st.divider()

    st.subheader("📥 Export Campaign Report")

    csv = campaigns.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download Campaign Report",
        data=csv,
        file_name="campaign_report.csv",
        mime="text/csv"
    )

st.caption(
    "Campaign records are stored locally using SQLite."
)