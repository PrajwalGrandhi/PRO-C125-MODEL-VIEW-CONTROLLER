from flask import Flask,request,jsonify
from clas import imagePredict

app=Flask(__name__)

@app.route('/pred-alphabet',methods=['POST'])
def predictImage():
    image=request.files.get("alphabet")
    prediction=imagePredict(image)
    return jsonify({
        'prediction':prediction
    },200)

if __name__ == "__main__":
    app.run(debug=True)