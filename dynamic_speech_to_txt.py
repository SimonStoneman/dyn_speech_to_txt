import whisper

model = whisper.load_model("base")  # or "small", "medium", "large"
result = model.transcribe("./hello_my_name_is_simon.wav")
print(result["text"])
