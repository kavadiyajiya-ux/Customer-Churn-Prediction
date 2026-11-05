
def generate_recommendations(customer):
    """
    Generate retention recommendations based on customer details.
    customer should be a pandas Series or dictionary.
    """

    recommendations = []

    contract = str(customer.get("Contract", ""))
    tenure = float(customer.get("tenure", 0) or 0)
    monthly_charges = float(customer.get("MonthlyCharges", 0) or 0)
    tech_support = str(customer.get("TechSupport", ""))
    online_security = str(customer.get("OnlineSecurity", ""))
    payment_method = str(customer.get("PaymentMethod", ""))

    # 1. Contract recommendation
    if contract == "Month-to-month":
        recommendations.append({
            "action": "Offer a long-term contract",
            "reason": "Customer currently has a month-to-month contract.",
            "priority": "High"
        })

    # 2. New customer recommendation
    if tenure <= 12:
        recommendations.append({
            "action": "Provide onboarding support",
            "reason": "Customer has been with the company for 12 months or less.",
            "priority": "High"
        })

    # 3. Technical support recommendation
    if tech_support == "No":
        recommendations.append({
            "action": "Offer a technical support package",
            "reason": "Customer does not currently have technical support.",
            "priority": "Medium"
        })

    # 4. Online security recommendation
    if online_security == "No":
        recommendations.append({
            "action": "Offer an online security package",
            "reason": "Customer does not currently have online security.",
            "priority": "Medium"
        })

    # 5. High monthly charges recommendation
    if monthly_charges >= 80:
        recommendations.append({
            "action": "Review the customer's pricing plan",
            "reason": "Customer has relatively high monthly charges.",
            "priority": "High"
        })

    # 6. Payment method recommendation
    if payment_method == "Electronic check":
        recommendations.append({
            "action": "Encourage automatic payment",
            "reason": "Customer currently uses electronic check payments.",
            "priority": "Low"
        })

    if not recommendations:
        recommendations.append({
            "action": "Maintain regular customer engagement",
            "reason": "No specific retention issue was identified by the rules.",
            "priority": "Low"
        })

    return recommendations
