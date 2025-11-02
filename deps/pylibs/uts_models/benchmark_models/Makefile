
all: clean run_all md

run_all:
	python cnn/examples/examples.py
	python hbos/examples/hbos.py
	python ae/examples/examples.py
	python generate_md.py
	python lof/examples/lof.py
	python vae/examples/examples.py
	python lstm/examples/examples.py
	python pca/examples/pca.py
	python random_forest/examples/examples.py
	python iforest/examples/iforest.py
	python ocsvm/examples/examples.py
	python svm/examples/examples.py



clean:
	python operate.py --type clear_imgs


md:
	pip install markdown2pdf pdfkit
	python generate_md.py

pdf:
	python operate.py --type pdf --arg1 "views.md" --arg2 "views.pdf"
