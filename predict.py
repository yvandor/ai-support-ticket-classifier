import joblib

type_model = joblib.load("ticket_model.joblib")
priority_model = joblib.load("priority_model.joblib")

print("Type a support ticket (type 'q' to quit)")

while True:
    text = input("\nTicket> ").strip()
    if text.lower() == "q":
        print("👋 Exiting")
        break

    # ----- TYPE -----
    type_probs = type_model.predict_proba([text])[0]
    type_labels = type_model.classes_
    type_top2 = type_probs.argsort()[-2:][::-1]

    print("\n📌 Predicted TYPE (top 2):")
    for i in type_top2:
        print(f"➡️ {type_labels[i]}: {type_probs[i]:.2f}")

    # ----- PRIORITY -----
    pr_probs = priority_model.predict_proba([text])[0]
    pr_labels = priority_model.classes_
    pr_top2 = pr_probs.argsort()[-2:][::-1]

    print("\n🚨 Predicted PRIORITY (top 2):")
    for i in pr_top2:
        print(f"➡️ {pr_labels[i]}: {pr_probs[i]:.2f}")
