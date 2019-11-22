# Required Imports
import os
from flask import Flask, request, jsonify
from firebase_admin import credentials, firestore, initialize_app
from flask_restful import Resource, Api, reqparse
from flask_cors import CORS
from itertools import chain
import pandas as pd
from pythainlp.tokenize import word_tokenize
import pickle
# Initialize Flask App
app = Flask(__name__)
CORS(app)

#losd model
modelMLP = pickle.load(open('MLP_model.plk', 'rb'))
modelNBmul = pickle.load(open('NBMUL_model.plk', 'rb'))
function_result = pd.read_csv('pp_dataframe.csv')
# Initialize Firestore DB
cred = credentials.Certificate('key.json')
default_app = initialize_app(cred)
db = firestore.client()
todo_ref = db.collection('history')


@app.route('/add', methods=['POST'])
def create():
    """
        e.g. json={'id': '1', 'title': 'Write a blog post','text':'ตัดคำกันหน่อยไหม'}
    """
    try:
        
        ttest = request.json['text']
        
        t2 = {ttest}
        artest = [t2]
        dftest = pd.DataFrame(artest,columns=["text"])

        tokenized_test = list([word_tokenize(l, engine="deepcut") for l in dftest.text])

        datalist2 = {}
        lishdata2 = []
        for r, tokens in enumerate(tokenized_test):
            #print(r,tokens)
            for i,value in enumerate(tokens):
                datalist2.update({value:1})
            lishdata2.append(datalist2)
            datalist2 = {}

        txttest = pd.DataFrame(lishdata2)

        for col_idx in function_result:
            for idxOftxt in txttest:  
                if col_idx == idxOftxt :
                    function_result[col_idx]=1

      
        
        predictionsMLP = modelMLP.predict(function_result)
        predisMLP = predictionsMLP[0]

        predictionsNBmul = modelNBmul.predict(function_result)
        predisNBmul = predictionsNBmul[0]
        
        id = request.json['id']
        todo_ref.document(str(id)).set(request.json)
        return jsonify({"success": True,"predictionsMLP":str(predisMLP),"predictionsNBmul":str(predisNBmul),"text":str(ttest)}), 200
    except Exception as e:
        return f"An Error Occured: {e}"

@app.route('/list', methods=['GET'])
def read():
    """
        all_todos : Return all documents
    """
    try:
        # Check if ID was passed to URL query
        todo_id = request.args.get('id')    
        if todo_id:
            todo = todo_ref.document(todo_id).get()
            return jsonify(todo.to_dict()), 200
        else:
            all_todos = [doc.to_dict() for doc in todo_ref.stream()]
            return jsonify(all_todos), 200
    except Exception as e:
        return f"An Error Occured: {e}"


@app.route('/update', methods=['POST', 'PUT'])
def update():
    """
         json={'id': '1', 'title': 'Write a blog post today'}
    """
    try:
        id = request.json['id']
        todo_ref.document(id).update(request.json)
        return jsonify({"success": True}), 200
    except Exception as e:
        return f"An Error Occured: {e}"


@app.route('/delete', methods=['GET', 'DELETE'])
def delete():
    """
        delete() : Delete a document from Firestore collection
    """
    try:
        # Check for ID in URL query
        todo_id = request.args.get('id')
        todo_ref.document(todo_id).delete()
        return jsonify({"success": True}), 200
    except Exception as e:
        return f"An Error Occured: {e}"


port = int(os.environ.get('PORT', 8080))
if __name__ == '__main__':
    app.run(threaded=True, host='0.0.0.0', port=port)