set title "Heat Map generated from a file containing Z values only"
unset key
set size 1,1
set tics in

# Color runs from white to green
set palette rgbformula -7,2,-7
set cbrange [0:5]
set cblabel "1234123s"
set grid
set border 4
$map1 << EOD
5 4 3 1 0
2 2 0 0 1
0 0 0 1 0
0 0 0 2 3
0 1 2 4 3
EOD

set view map
splot '$map1' matrix with image