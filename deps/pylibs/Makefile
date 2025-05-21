test:
   # before we run experiment, we need to run this test
	python -m unittest discover -s ./pylibs/test

debug:
	python -m  unittest -v pylibs/test/test_torch.py

clearn_cache:
	git rm -r --cached .
	git add --all
	git commit -m "clear ignore file"

p:
	git add --all
	git commit -m "update"
	git push origin dev