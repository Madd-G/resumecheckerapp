from flask import Flask, flash, render_template, request, redirect, url_for
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config["DEBUG"] = False

# Allowed files
ALLOWED_EXTENSIONS = {'pdf'}

UPLOAD_FOLDER = 'static/files/'


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/try/')
def coba():
    return render_template('try.html')

@app.route("/", methods=['GET', 'POST'])
def upload_file():
    print("======= upload_file() ======")
    inp = request.form.get('kualifikasi')
    print(inp)
    if request.method == 'GET':
        return render_template('index.html')
    else:
        if 'file' not in request.files:
            flash('No File part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            print("==== tidak ada file yang dipilih ====")
            flash('Tidak ada file yang dipilih !')
            return redirect(request.url)
        if not allowed_file(file.filename):
            print("==== format file tidak tepat ====")
            flash('Masukan file gambar dengan tipe extensi ' + str(ALLOWED_EXTENSIONS))
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            print("==== ada file dan format tepat ====")
            # return redirect(url_for('result_text', filename=filename))
            return result_text(filename, inp)


@app.route('/index/<filename>/scaned', methods=['GET', 'POST'])
def result_text(filename, inp):
    inp = inp
    path = (UPLOAD_FOLDER + filename)
    from resume import resume_classification
    text, score_str, cosine = resume_classification(path, inp)
    results = []
    answer = "<div class='col text-center'>" "Klasifikasi: " + text + " dengan score : " + score_str + ", cosine score: " + str(cosine) + "%" + "</div>"
    results.append(answer)
    return render_template('index.html', len=len(results), results=results)


app.config['SECRET_KEY'] = 'super secret key'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB upload limit
if __name__ == '__main__':
    app.run()
    # app.run(debug=True, use_reloader=True)
