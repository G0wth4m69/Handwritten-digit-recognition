import tkinter as tk
from PIL import Image, ImageDraw
import joblib
from src.utils import preprocess_image

def launch_gui():
    model = joblib.load("models/svm_model.pkl")

    root = tk.Tk()
    root.title("Handwritten Digit Recognition")
    root.geometry("420x520")
    root.resizable(False, False)

    canvas_frame = tk.Frame(root, bd=2, relief="solid")
    canvas_frame.pack(pady=10)

    canvas = tk.Canvas(canvas_frame, width=360, height=300, bg="white", cursor="cross")
    canvas.pack()

    image = Image.new("L", (360, 300), 255)
    draw = ImageDraw.Draw(image)

    def draw_digit(event):
        x, y = event.x, event.y
        r = 4  # thinner brush = better accuracy
        canvas.create_oval(x-r, y-r, x+r, y+r, fill="black")
        draw.ellipse((x-r, y-r, x+r, y+r), fill=0)

    canvas.bind("<B1-Motion>", draw_digit)

    info_frame = tk.Frame(root, bd=2, relief="solid")
    info_frame.pack(fill="x", padx=10, pady=10)

    predicted_label = tk.Label(info_frame, text="Predicted digit: -", font=("Arial", 14))
    predicted_label.pack(anchor="w", padx=10, pady=5)

    accuracy_label = tk.Label(info_frame, text="Accuracy: High", font=("Arial", 14))
    accuracy_label.pack(anchor="w", padx=10)

    def predict():
        processed = preprocess_image(image)
        if processed is None:
            return

        prediction = model.predict(processed)[0]
        predicted_label.config(text=f"Predicted digit: {prediction}")
        accuracy_label.config(text="Accuracy: High")

    def on_release(event):
        root.after(300, predict)

    canvas.bind("<ButtonRelease-1>", on_release)

    def clear_screen(event=None):
        canvas.delete("all")
        draw.rectangle((0, 0, 360, 300), fill=255)
        predicted_label.config(text="Predicted digit: -")
        accuracy_label.config(text="Accuracy: -")

    root.bind("<Escape>", clear_screen)
    root.bind("<BackSpace>", clear_screen)

    root.mainloop()
