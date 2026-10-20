def linear_regression(xs, ys):
    n = len(xs)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    b = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / sum((x - mean_x) ** 2 for x in xs)
    a = mean_y - b * mean_x
    return a, b

def predict(a, b, x):
    return a + b * x

def main():
    months = [1, 2, 3]
    weights = [18.0, 19.0, 20.5]   # kg (dummy data)
    next_month = 4
    threshold = 21.0               # kg (dummy upper limit, replace with the official Z-score cutoff)

    a, b = linear_regression(months, weights)
    prediction = predict(a, b, next_month)

    print(f"Model      : y = {a:.2f} + {b:.2f}x")
    print(f"Prediction : month {next_month} -> {prediction:.2f} kg")
    if prediction > threshold:
        print("Status     : EARLY WARNING - projected weight is above the threshold")
    else:
        print("Status     : Normal trend")

if __name__ == "__main__":
    main()
