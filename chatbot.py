from intent_module import detect_intent
from faq_module import faq_module
from recommendation_module import recommend_products
from product_module import product_question
from review_module import review_module


def main():

    print("=" * 60)
    print("             E-COMMERCE AI CHATBOT")
    print("=" * 60)

    print("Type 'exit' to stop the chatbot.")


    while True:

        query = input("\nYou: ")


        if query.lower() == "exit":

            print("\nChatbot: Thank you for using the E-Commerce AI Chatbot!")
            break


        # -----------------------------
        # INTENT DETECTION
        # -----------------------------

        intent = detect_intent(query)

        print("\nDetected Intent:", intent)


        # -----------------------------
        # FAQ
        # -----------------------------

        if intent == "FAQ":

            response = faq_module(query)

            if response:

                print("\nAssistant:")
                print(response)

            else:

                print("\nAssistant:")
                print(
                    "I could not find a relevant FAQ "
                    "for your question."
                )


        # -----------------------------
        # RECOMMENDATION
        # -----------------------------

        elif intent == "RECOMMENDATION":

            response = recommend_products(query)

            print("\nAssistant:")
            print(response)


        # -----------------------------
        # PRODUCT
        # -----------------------------

        elif intent == "PRODUCT":

            response = product_question(query)

            print("\nAssistant:")
            print(response)


        # -----------------------------
        # REVIEW
        # -----------------------------

        elif intent == "REVIEW":

            product_name = input(
                "Enter the exact product name: "
            )

            print("\nReview Analysis:")

            review_module(product_name)


if __name__ == "__main__":
    main()