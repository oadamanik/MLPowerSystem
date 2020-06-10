import pandapower as pp


def test_network():
    net = pp.create_empty_network()

    bus_voltage_kv = 110
    bus_1 = pp.create_bus(net, bus_voltage_kv, name='Bus 1')
    bus_2 = pp.create_bus(net, bus_voltage_kv, name='Bus 2')
    bus_3 = pp.create_bus(net, bus_voltage_kv, name='Bus 3')
    bus_4 = pp.create_bus(net, bus_voltage_kv, name='Bus 4')
    bus_5 = pp.create_bus(net, bus_voltage_kv, name='Bus 5')
    bus_6 = pp.create_bus(net, bus_voltage_kv, name='Bus 6')
    bus_7 = pp.create_bus(net, bus_voltage_kv, name='Bus 7')
    bus_8 = pp.create_bus(net, bus_voltage_kv, name='Bus 8')
    bus_9 = pp.create_bus(net, bus_voltage_kv, name='Bus 9')

    gen_1_p_mw = 0
    gen_2_p_mw = 163
    gen_3_p_mw = 85

    pp.create_gen(net, bus_1, gen_1_p_mw, slack=True, name='Gen 1')
    pp.create_sgen(net, bus_2, gen_2_p_mw, name='Gen 2')
    pp.create_sgen(net, bus_3, gen_3_p_mw, name='Gen 3')

    load_5_p_mw = 90
    load_5_q_mvar = 30
    load_7_p_mw = 100
    load_7_q_mvar = 35
    load_9_p_mw = 125
    load_9_q_mvar = 50

    pp.create_load(net, bus_5, load_5_p_mw, load_5_q_mvar, name='Load 5')
    pp.create_load(net, bus_7, load_7_p_mw, load_7_q_mvar, name='Load 7')
    pp.create_load(net, bus_9, load_9_p_mw, load_9_q_mvar, name='Load 9')

    line_length = 10
    pp.create_line(net, bus_1, bus_4, line_length, '184-AL1/30-ST1A 110.0', name='Line 1-4')
    pp.create_line(net, bus_2, bus_8, line_length, '184-AL1/30-ST1A 110.0', name='Line 2-8')
    pp.create_line(net, bus_3, bus_6, line_length, '184-AL1/30-ST1A 110.0', name='Line 3-6')
    pp.create_line(net, bus_4, bus_5, line_length, '184-AL1/30-ST1A 110.0', name='Line 4-5')
    pp.create_line(net, bus_4, bus_9, line_length, '184-AL1/30-ST1A 110.0', name='Line 4-9')
    pp.create_line(net, bus_5, bus_6, line_length, '184-AL1/30-ST1A 110.0', name='Line 5-6')
    pp.create_line(net, bus_6, bus_7, line_length, '184-AL1/30-ST1A 110.0', name='Line 6-7')
    pp.create_line(net, bus_7, bus_8, line_length, '184-AL1/30-ST1A 110.0', name='Line 7-8')
    pp.create_line(net, bus_8, bus_9, line_length, '184-AL1/30-ST1A 110.0', name='Line 8-9')

    return net