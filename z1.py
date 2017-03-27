from math import sqrt
numbers = [1,2,3,4,5]

def avg(numbers):
	sum = 0
	count = 0
	for val in numbers:
	    sum += val
	    count += 1
	return (sum/count,count)


def variance(numbers):
	total = 0
	count
	avg_tuple = avg(numbers)
	for val in numbers:
		total += (val - avg_tuple[0]) ** 2
	return total/avg_tuple[1]

def std(numbers):
	return sqrt(variance(numbers))

print str(avg(numbers)[0])
