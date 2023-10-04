from strongsort import StrongSORT

tracker = StrongSORT(model_weights='model.pth', device='cuda')
pred = model(img)
for i, det in enumerate(pred):
    det[i] = tracker[i].update(detection, im0s)

class StrongSortTracker:
    def __init__(self):
        tracker = StrongSORT(model_weights='model.pth', device='cuda')

    def update_all(self, preds):
        for i, det in enumerate(pred):

    def update(self, pred):

            det[i] = tracker[i].update(detection, im0s)
