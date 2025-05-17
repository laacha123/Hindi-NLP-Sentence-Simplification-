from transformers import T5Tokenizer, T5ForConditionalGeneration

# Load model and tokenizer
model_path = "./trained_model"
model = T5ForConditionalGeneration.from_pretrained(model_path).to(device)
tokenizer = T5Tokenizer.from_pretrained(model_path)

# Run prediction
def simplify_text(text):
    input_ids = tokenizer.encode(f"simplify: {text}", return_tensors="pt", max_length=128, truncation=True).to(device)
    output_ids = model.generate(input_ids, max_length=128, num_beams=4, early_stopping=True)
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

test_sentences = [
    "जब प्रधानमंत्री ने संसद में आर्थिक सुधारों की घोषणा की, तो विपक्ष ने इसका विरोध करते हुए तीखी बहस छेड़ दी और सदन की कार्यवाही को बाधित कर दिया।",
    "जैसे ही अंतरिक्ष यान ने पृथ्वी की कक्षा में प्रवेश किया, वैज्ञानिकों ने मिशन की सफलता का जश्न मनाया और अगले चरण की तैयारियों में जुट गए।",
    "भारी वर्षा के कारण नदियाँ उफान पर आ गईं, जिससे कई गांवों में बाढ़ आ गई और लोगों को सुरक्षित स्थानों पर भेजना पड़ा।",
    "विद्यालय प्रशासन द्वारा परीक्षा की नई प्रणाली लागू किए जाने के बाद छात्रों और अभिभावकों में भ्रम की स्थिति पैदा हो गई, जिससे कई बार मीटिंग्स आयोजित करनी पड़ीं।",
    "जैसे ही अंतरराष्ट्रीय सम्मेलन में भारत के प्रतिनिधि ने भाषण देना शुरू किया, दुनिया भर के प्रतिनिधियों ने ध्यानपूर्वक उनकी बातों को सुना और तालियों की गूंज से सभा स्थल गूंज उठा।"
]

for sentence in test_sentences:
    simplified = simplify_text(sentence)
    print(f"\nOriginal: {sentence}")
    print(f"Simplified: {simplified}")
