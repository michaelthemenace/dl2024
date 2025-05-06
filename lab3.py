import math


def predict(x1i, x2i, w1, w2, w0):
    return w1 * x1i + w2 * x2i + w0


def cal_single_loss(yi, yi_pred):
    return -(
        yi * math.log(1 + math.exp(-yi_pred))
        + (1 - yi) * (-yi_pred - math.log(1 + math.exp(-yi_pred)))
    )


def cal_new_w0(w0, r, yi, yi_pred):
    return w0 - r * (1 - yi - 1 / -(1 + math.exp(-yi_pred)))


def cal_new_w1(w1, r, x1i, yi, yi_pred):
    return w1 - r * (-yi * x1i - x1i * (1 - 1 / -(1 + math.exp(-yi_pred))))


def cal_new_w2(w2, r, x2i, yi, yi_pred):
    return w2 - r * (-yi * x2i - x2i * (1 - 1 / -(1 + math.exp(-yi_pred))))


def logistic_regression(file_path, r=0.5, w0=0, w1=0, w2=0):

    with open(file_path, "r") as file:
        lines = file.readlines()

    n = len(lines) - 1
    total_loss = 0

    for i in range(1, n):
        x1i, x2i, yi = map(float, lines[i].strip().split(","))
        yi_pred = predict(x1i, x2i, w1, w2, w0)
        w0 = cal_new_w0(w0, r, yi, yi_pred)
        w1 = cal_new_w1(w1, r, x1i, yi, yi_pred)
        w2 = cal_new_w2(w2, r, x2i, yi, yi_pred)
        Li = cal_single_loss(yi, yi_pred)
        print(f"{i}th iteration:")
        print(f"w0: {w0}, w1: {w1}, w2: {w2}, yi_pred: {yi_pred} loss: {Li}")
        total_loss += Li

    final_loss = total_loss / n
    print(f"Final w0: {w0}, Final w1: {w1}, Final w2: {w2}, Final loss: {final_loss}")

    return final_loss


if __name__ == "__main__":
    file_path = "/home/mike/Documents/USTH/Deep-Learning/dl2024/loan.csv"
    logistic_regression(file_path)
