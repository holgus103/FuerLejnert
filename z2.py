from random import Random

def belongs_to_circle(x, y):
	return ((x - 0.5)**2 + (y - 0.5)**2) < 0.25
gen = Random()
no_elements = 1000
count = 0
for i in range(0, no_elements):
	x = gen.random()
	y = gen.random()
	if belongs_to_circle(x,y):
		count += 1
area = 1.0 * count/no_elements
print "area estimation"
print count
print 4*area
