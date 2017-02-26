import binary
assert binary.zbits(4, 3) == {'0100', '0001', '0010', '1000'}
assert binary.zbits(4, 1) == {'0111', '1011', '1101', '1110'}
assert binary.zbits(5, 4) == {'00001', '00100', '01000', '10000', '00010'}