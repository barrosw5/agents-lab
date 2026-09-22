import random as r

generic_messages = ["Keep going, please.", "I see.", "Tell me more about that."]

while True:
    user_input = input("You: ").strip()

    if "idiot" in user_input.lower():
        print("ELIZA: WHY")
    elif "I feel" in user_input:
        print("ELIZA: " + user_input.replace("I feel", "Why do you feel") + "?")
    else:
        print("ELIZA: " + r.choice(generic_messages))
