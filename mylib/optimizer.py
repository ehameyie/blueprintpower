from __future__ import division
from pyomo.environ import *
import pyomo.opt as po
import logging
import pandas as pd
import pyomo.opt as po
import numpy as np

__author__ = 'Eunice Hameyie'
__version__ = 'v1.0'

'''
This is my main class library to dispatch a battery for market participation using pyomo.

Version Control:
v1.0 - Discharge battery based on constraints for the Blueprint Power Challenge - published on Nov 7 2018
'''


class batdispatch(object):
    ''' A class to dispatch a battery charged by the grid. Dispatch is per following requirements as per the Blueprint Power Challenge.
    Overall System Requirements  :
        1. The system SHALL optimize the battery storage dispatch (with an optimization time horizon of at
        least 1 day) for the day ahead energy market
            ○ The battery storage’s State of Energy SHALL be continuous between optimization time
            horizon boundaries
        2. The system SHALL accept the following as inputs for the battery storage asset:
            ○ Max discharge power capacity (kW)
            ○ Max charge power capacity (kW)
            ○ Discharge energy capacity (kWh)
            ○ AC-AC Round-trip efficiency (%)
            ○ Maximum daily discharged throughput (kWh)
        3. The system SHALL accept the following as inputs for the market revenues:
            ○ Hourly LBMP ($/MWh)
            ○ Zone
        4. The system SHALL output the following values about a given battery storage system, for a year’s
        worth of data, at an hourly resolution
            ○ Power output (kW)
            ○ State of Energy (kWh)
        5. The system SHALL output the following summary values about a given storage system:
            ○ Total annual revenue generation ($)
            ○ Total annual charging cost ($)
            ○ Total annual discharged throughput (kWh)
        6. The system SHALL output the following plots
            ○ A plot that includes both hourly battery dispatch and hourly LBMP for the most
            profitable week
            ○ A plot that shows the total profit for each month

    Inputs:
    self.df = input dataframe with market data
    self.eff_bat = AC-AC Round-trip efficiency (%)
    self.bat_kw_max_ch = Max charge power capacity (kW
    self.bat_kw_max_disch = Max discharge power capacity (kW
    self.bat_kwh_max = Discharge energy capacity (kWh)
    self.bat_kwh_max_day = Maximum daily discharged throughput (kWh)

    Outputs:
    Date : timestamp
    battery dispatch_CHARGE (kW)': battery charge in kW
    battery dispatch_DISCHARGE (kW) : battery discharge in kW
    Power output (kW) = Power output in kW
    State of Energy (kWh) : current state of battery energy (State of Energy (kWh))

    '''

    def __init__(self, df, eff_bat, bat_kw_max_ch, bat_kw_max_disch, bat_kwh_max, bat_kwh_max_day):
        """Return a connector object whose attributes are x, y, z, v, u and w"""
        self.df = df #input dataframe
        self.eff_bat = eff_bat
        self.bat_kw_max_ch = bat_kw_max_ch
        self.bat_kw_max_disch = bat_kw_max_disch
        self.bat_kwh_max = bat_kwh_max
        self.bat_kwh_max_day = bat_kwh_max_day

        node_data = self.df.copy()
        self.size_node = len(node_data)
        shape_node = len(node_data)
        self.ones_node = np.ones(self.size_node)

        node_data['lower_bound'] = -1.0*self.bat_kwh_max_day
        node_data['upper_bound'] = 1.0*self.bat_kwh_max_day
        node_data['initialize'] = 0.0
        node_data['capacity_bat_max'] = self.bat_kwh_max
        node_data['capacity_bat_min'] = 0.0
        node_data['bat_kW_max'] = self.bat_kw_max_ch/self.eff_bat
        node_data['bat_kW_min'] = -1.0*self.bat_kw_max_disch/self.eff_bat

        self.node_data = node_data

        #Meter interval
        self.interval= (self.node_data.Date[1] - self.node_data.Date[0]).seconds/60.0/60.0
        self.interval_inv = 1.0/self.interval
        self.interval_min = self.interval * 60


    def drMarket(self):
        ''' Optimize battery for DR market'''
        #Sets
        node_set = self.node_data.index.copy()

        #Parameters
        rev_dr = self.node_data['LBMP ($/MWHr)'].copy()

        #initialize
        initial_values = self.node_data.copy()['initialize'].to_dict()

        #Model for optimization
        model = ConcreteModel()

        #bounds
        lb_cap = self.node_data['capacity_bat_min']
        ub_cap = self.node_data['capacity_bat_max']
        lb_q = 0 * self.node_data['bat_kW_min']
        ub_q = self.node_data['bat_kW_max']
        lb_q_z = self.node_data['bat_kW_min']
        ub_q_z = 0 * self.node_data['bat_kW_max']
        lb_cap_day = self.node_data['lower_bound']
        ub_cap_day = self.node_data['upper_bound']

        def f_xbound(m, i):
            return (lb_q[i], ub_q[i])

        def f_ybound(m, i):
            return (lb_cap[i], ub_cap[i])

        def f_zbound(m, i):
            return (lb_q_z[i], ub_q_z[i])

        def init_conditions(m):
            yield m.x[0] == 0
            yield m.y[0] == 0
            yield m.z[0] == 0

        # Variables
        model.x = Var(node_set, domain=NonNegativeReals, bounds = f_xbound)
        model.y = Var(node_set, domain=NonNegativeReals, bounds = f_ybound)
        model.z = Var(node_set, domain=Reals, bounds = f_zbound)

        model.init_conditions=ConstraintList(rule=init_conditions)

        #Objective function
        model.obj = Objective(expr = -1 * sum(self.eff_bat*(model.x[i] + model.z[i])*self.node_data.ix[i,'LBMP ($/MWHr)'] for i in node_set) , sense = maximize)


        def capday_rule(model):
            return (lb_cap_day[0], sum(self.eff_bat*model.z[i]*self.interval for i in node_set), ub_cap_day[0])
        model.capday = Constraint(rule = capday_rule)

        def cap_rule(model, i):
            if i != 0:
                return model.y[i] == model.y[i-1] + self.eff_bat * (model.x[i] + model.z[i]) * self.interval
            else:
                return Constraint.Skip
        model.cap = Constraint(node_set, rule = cap_rule)

        #Solve
        def solve(m):
            """Solve the model."""
            solver = po.SolverFactory('glpk')
            results = solver.solve(m)#, tee=True, keepfiles=False, options_string="mip_tolerances_integrality=1e-9 mip_tolerances_mipgap=0")

            if (results.solver.status != pyomo.opt.SolverStatus.ok):
                logging.warning('Check solver not ok?')
            if (results.solver.termination_condition != pyomo.opt.TerminationCondition.optimal):
                logging.warning('Check solver optimality?')

            return results

        results = solve(model)
        results.write()

        #Output for graphing
        battery_dispatch_ch = []
        battery_state = []
        battery_dispatch_disch = []
        # accumulated_disch = []

        for i in node_set:
           battery_dispatch_ch.append(value(model.x[i]))
           battery_state.append(value(model.y[i]))
           battery_dispatch_disch.append(value(model.z[i]))
           # accumulated_disch.append(value(model.z))

        out = pd.concat([self.node_data['Date'], pd.DataFrame(battery_dispatch_ch, columns = ['battery dispatch_CHARGE (kW)']), \
                        pd.DataFrame(battery_state, columns = ['State of Energy (kWh)']), \
                        pd.DataFrame(battery_dispatch_disch, columns = ['battery dispatch_DISCHARGE (kW)']), \
                        self.node_data['LBMP ($/MWHr)']], axis = 1)
        print(results)
        out['Power output (kW)'] = out['battery dispatch_CHARGE (kW)'] + out['battery dispatch_DISCHARGE (kW)']
        return out[['Date', 'LBMP ($/MWHr)', 'Power output (kW)', 'State of Energy (kWh)']]
