input=$1
base=$(basename $input)
input=_posts/$base
#if [ ! -f ./_posts/$base ]; then
cp $1 ./_posts
#fi

git branch svgs # if this isn't already there
#python -m readme2tex --output READMETMP.md $input --username dadashkarimi --branch master --project dadashkarimi.github.io --usepackage amsmath --usepackage amssymb --usepackage caption
python -m readme2tex --output READMETMP.md $input --username dadashkarimi --branch svgs --project dadashkarimi.github.io --usepackage amsmath
mv READMETMP.md $input
git add $input
git add svgs
git commit -a -m "adding new post"
git push
#git push origin HEAD:master

