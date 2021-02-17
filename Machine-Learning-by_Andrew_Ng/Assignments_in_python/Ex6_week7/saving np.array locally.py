a = np.array([1, 2, 3, 4])
np.savetxt('test1.txt', a, fmt='%d')
b = np.loadtxt('test1.txt', dtype=int)
a == b

a.tofile('test2.dat')
c = np.fromfile('test2.dat', dtype=int)
c == a
# array([ True,  True,  True,  True], dtype=bool)

np.save('test3.npy', a)    # .npy extension is added if not given
d = np.load('test3.npy')
a == d
# array([ True,  True,  True,  True], dtype=bool)