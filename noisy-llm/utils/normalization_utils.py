
def trim_single_prediction(prediction):
    for line in prediction.split("\n"):
        if line!="":
            return line
    return ""

def trim_prediction(predictions):
    return [trim_single_prediction(prediction) for prediction in predictions]