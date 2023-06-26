from getpatelem import getclaim1
from flask import Flask, request, render_template, redirect
# from torchtext.data import Field, TabularDataset, BucketIterator
# from lstm import tokenize, LSTMClassifier
from distilbert import DistilbertClassifier, tokenizer_ditilbert
import torch
import torch.nn.functional as F
import pickle

# number = 'US6477427B1'
# claim_1st = getclaim1(number)
# net = DistilbertClassifier()
# net.load_state_dict(torch.load('./src/distilbert512.pt'))

net = DistilbertClassifier()
net.load_state_dict(torch.load('./distilbert512.pt'))
net.eval()

# net.eval()
# ids, mask = tokenizer_ditilbert(claim_1st)
    
# # 推論の実行
# with torch.no_grad():
#     y = net(ids.unsqueeze(0), mask.unsqueeze(0))
#     y = torch.argmax(F.softmax(y, dim=-1)).detach().numpy()

app = Flask(__name__)

@app.route('/', methods = ['GET', 'POST'])
def predicts():
    if request.method == 'POST':
        number = request.form['patentnumber']
        if number == '':
            return redirect(request.url)
        claim_1st = getclaim1(number)
        if claim_1st == 'not extracted':
            return render_template('result1.html', claim_1st = claim_1st, number = number)
        # net = DistilbertClassifier()
        # net.load_state_dict(torch.load('./distilbert512.pt'))
        # net.eval()
        ids, mask = tokenizer_ditilbert(claim_1st)
        # 推論の実行
        with torch.no_grad():
            y = net(ids.unsqueeze(0), mask.unsqueeze(0))
            y = torch.argmax(F.softmax(y, dim=-1)).detach().numpy()
        
        category = str(y)
        return render_template('result.html', claim_1st = claim_1st, number = number, category=category)
    elif request.method == 'GET':
        return render_template('index.html')
        
if __name__ == '__main__':
    app.run()

