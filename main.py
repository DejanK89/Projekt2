from neural_network import NeuralNetwork

def build_dataset_4():
    inputs = []
    targets = []

    for i in range(16):
        row = [
            (i >> 3) & 1,
            (i >> 2) & 1,
            (i >> 1) & 1,
            i & 1
        ]
        inputs.append(row)
        targets.append(1 if sum(row) % 2 == 1 else 0)

    return inputs, targets

if __name__ == "__main__":
    inputs, targets = build_dataset_4()

    #net = NeuralNetwork(num_inputs=4, num_hidden=8, learning_rate=0.2, seed=1)
    #net.train(inputs, targets, epochs=50000, print_every=10000)
    print("=== Inställningar för nätverket ===")

    num_hidden = int(input("Antal noder i hidden layer: "))
    learning_rate = float(input("Learning rate: "))
    epochs = int(input("Antal epoker: "))

    net = NeuralNetwork(
    num_inputs=4,
    num_hidden=num_hidden,
    learning_rate=learning_rate,
    seed=1
)

    net.train(inputs, targets, epochs=epochs, print_every=max(1, epochs // 5))

    print("\nTest efter träning:")
    correct = 0
    for x, t in zip(inputs, targets):
        out = net.predict_one(x)
        pred = 1 if out > 0.5 else 0
        correct += (pred == t)
        print(f"inputs={x} target={t} output={out:.2f} predicted={pred}")

    print(f"\nAccuracy: {correct}/16")
    print("\n--- WEIGHTS TO COPY ---")
    print("hidden_weights =", net.hidden_weights)
    print("hidden_bias =", net.hidden_bias)
    print("output_weights =", net.output_weights)
    print("output_bias =", net.output_bias)
#-----------------------------------------------------------------------------
print("\nLive-demo: skriv fyra bitar, t.ex. 1011. Skriv q för att avsluta.")

last_input = None

while True:
    user_input = input("Ange 4 bitar: ").strip()

    if user_input.lower() == "q":
        print("Avslutar demo.")
        break

    if len(user_input) != 4 or any(ch not in "01" for ch in user_input):
        print("Fel input. Skriv exakt fyra bitar, t.ex. 0101.")
        continue

    bits = [int(ch) for ch in user_input]

    if bits != last_input:
        output = net.predict_one(bits)
        prediction = net.predict_label(bits)

        led_state = "ON" if prediction == 1 else "OFF"

        print(f"Input: {bits}")
        print(f"Output: {output:.4f}")
        print(f"Prediction: {prediction}")
        print(f"LED: {led_state}")

        last_input = bits
    else:
        print("Ingen förändring i insignaler.")
    




    









         