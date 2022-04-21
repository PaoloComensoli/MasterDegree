\* MaximizeProfit *\
Maximize
OBJ: 3500 TRANSPORT_('Cargo_1',_'Wagon_1')
 + 3500 TRANSPORT_('Cargo_1',_'Wagon_2')
 + 3500 TRANSPORT_('Cargo_1',_'Wagon_3')
 + 3500 TRANSPORT_('Cargo_1',_'Wagon_4')
 + 2500 TRANSPORT_('Cargo_2',_'Wagon_1')
 + 2500 TRANSPORT_('Cargo_2',_'Wagon_2')
 + 2500 TRANSPORT_('Cargo_2',_'Wagon_3')
 + 2500 TRANSPORT_('Cargo_2',_'Wagon_4')
 + 2000 TRANSPORT_('Cargo_3',_'Wagon_1')
 + 2000 TRANSPORT_('Cargo_3',_'Wagon_2')
 + 2000 TRANSPORT_('Cargo_3',_'Wagon_3')
 + 2000 TRANSPORT_('Cargo_3',_'Wagon_4')
Subject To
_C1: TRANSPORT_('Cargo_1',_'Wagon_1') + TRANSPORT_('Cargo_1',_'Wagon_2')
 + TRANSPORT_('Cargo_1',_'Wagon_3') + TRANSPORT_('Cargo_1',_'Wagon_4') <= 20
_dummy: __dummy = 0
_C10: __dummy <= 8000
_C11: __dummy <= 2500
_C2: TRANSPORT_('Cargo_2',_'Wagon_1') + TRANSPORT_('Cargo_2',_'Wagon_2')
 + TRANSPORT_('Cargo_2',_'Wagon_3') + TRANSPORT_('Cargo_2',_'Wagon_4') <= 10
_C3: TRANSPORT_('Cargo_3',_'Wagon_1') + TRANSPORT_('Cargo_3',_'Wagon_2')
 + TRANSPORT_('Cargo_3',_'Wagon_3') + TRANSPORT_('Cargo_3',_'Wagon_4') <= 18
_C4: TRANSPORT_('Cargo_1',_'Wagon_1') + TRANSPORT_('Cargo_2',_'Wagon_1')
 + TRANSPORT_('Cargo_3',_'Wagon_1') <= 10
_C5: TRANSPORT_('Cargo_1',_'Wagon_2') + TRANSPORT_('Cargo_2',_'Wagon_2')
 + TRANSPORT_('Cargo_3',_'Wagon_2') <= 8
_C6: TRANSPORT_('Cargo_1',_'Wagon_3') + TRANSPORT_('Cargo_2',_'Wagon_3')
 + TRANSPORT_('Cargo_3',_'Wagon_3') <= 12
_C7: TRANSPORT_('Cargo_1',_'Wagon_4') + TRANSPORT_('Cargo_2',_'Wagon_4')
 + TRANSPORT_('Cargo_3',_'Wagon_4') <= 6
_C8: __dummy <= 5000
_C9: __dummy <= 4000
Bounds
 __dummy = 0
End
