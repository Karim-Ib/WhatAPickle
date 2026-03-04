"""
Prompt sets for CLIP zero-shot classification.

This is the main tuning lever. CLIP matches images against these text
descriptions, so the quality and diversity of prompts directly controls
detection accuracy.

Guidelines for adding prompts:
- Be specific and visual ("sliced pickles on a white plate" > "pickles")
- Cover different contexts: standalone, in dishes, in jars, memes
- Non-pickle prompts should include confusable items (cucumber, zucchini)
- Keep the two sets roughly balanced in count to avoid score bias
"""

PICKLE_PROMPTS = [
    # Direct
    "a photo of a pickle",
    "a photo of pickles",
    "a photo of pickled cucumbers",
    "a jar of pickles",
    "a plate with pickles",
    "pickles on a cutting board",
    "sliced pickles",
    "a pickle spear",
    "dill pickles",
    "gherkins",

    # Contextual — pickles visible in a dish
    "visible pickles on a burger",
    "pickles on the side of a plate",
    "a relish tray with pickles",
    "fried pickles",

    # Meme / internet context
    "a cartoon pickle",
    "pickle rick",
    "a funny pickle meme",
]

NON_PICKLE_PROMPTS = [
    # Generic food
    "a photo of food",
    "a photo of a meal",
    "a photo of vegetables",
    "a photo of a salad",
    "a photo of a snack",

    # Burgers/sandwiches — keep as anchors but less specific
    "a photo of a burger",
    "a photo of a sandwich",

    # Cucumber — hardest confusion case
    "a photo of a cucumber",
    "fresh cucumber slices",
    "a whole raw cucumber",
    "cucumber slices on a cutting board",
    "a cucumber salad",

    # Other green confusables
    "zucchini slices",
    "green peppers",
    "jalapeno peppers",

    # Non-food
    "a photo of a person",
    "a photo of a landscape",
    "a photo of an animal",
    "a selfie",
    "a screenshot",
    "a meme",
]