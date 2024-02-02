"""
parameters.py

Use this to manage parameters.

Each parameter has a dictionary with the following keys
    free : bool
        True means the parameter is free. False means the parameter is fixed
    value : float
        the actual value
    limits : list of 2 floats
        the limit for the parameter to exist
    tag : str
        the string for plotting
"""

def check_plan(plan):
    """
    iterate through the plan to see if the value is within limits
    """
    for ikey in plan.keys():
        if plan[ikey]['free']:
            lim = plan[ikey]['limits']
            if plan[ikey]['value'] < lim[0]:
                raise ValueError('The free parameter value is less than the lower limit, which is not internally consistent: %s'%ikey)
            if plan[ikey]['value'] > lim[1]:
                raise ValueError('The free parameter value is greater than the upper limit, which is not internally consistent: %s'%ikey)

class manager(object):
    def __init__(self, ):
        self.plan = {}
        self.parnames = [] # ordering of the parameters
        self.free_parameters = [] # ordering of the free parameters

    def add_plan(self, iplan):
        """
        add a dictionary to the plan

        Inputs
        ------
        iplan : dictionary
        """
        check_plan(iplan)

        self.plan.update(iplan)
        self.parnames = sorted(list(self.plan.keys()))

        self.get_free_parameters()

    def fill_default(self, default_par):
        """
        the user can provide a dictionary of default parameter values.
        These will all be fixed and not act as free parameters

        Arguments
        ---------
        default_par : dictionary
            the key name is the parameter name. Each entry is a dictionary with:
            'tag' = name of the parameter for printing
            'value' = value of the parameter
        """
        for ikey in default_par.keys():
            if ikey not in self.parnames:
                entry = {'free':True, 'limits':None}.update(default_par[ikey])

                self.plan[ikey] = entry
        self.parnames = sorted(list(self.plan.keys()))

    def free(self, ikey):
        """
        retrieve the input for 'free' for a certain key
        """
        return self.plan[ikey]['free']

    def value(self, ikey):
        return self.plan[ikey]['value']

    def limits(self, ikey):
        return self.plan[ikey]['limits']

    def tag(self, ikey):
        return self.plan[ikey]['tag']

    def get_free_parameters(self):
        """
        get a list of names of the free parameters
        """
        free_key = []
        free = [self.plan[ikey]['free'] for ikey in self.parnames]

        for i, ikey in enumerate(self.parnames):
            if self.plan[ikey]['free']:
                free_key.append(ikey)

        self.free_parameters = free_key

    def get_value_dict(self, ptheta):
        """
        provide a dictionary of the parameters with values that have the updated values from dynesty
        Arguments
        ---------
        ptheta : list of float
            the current sample of values in dynesty. The length should be the same as self.free_parameters

        """
        # check
        if len(ptheta) != len(self.free_parameters):
            raise ValueError('inconsistency between input values and the expected free parameters')

        par = {}
        for ikey in self.parnames:
            try:
                i = self.free_parameters.index(ikey)
                par[ikey] = ptheta[i]
            except:
                par[ikey] = self.plan[ikey]['value']

        return par

    def value_of_free_parameters(self):
        """
        return the values of the free parameters
        Return
        ------
        list
            the list of values should be in the same order as free_parameters
        """
        vals = []
        for ikey in self.free_parameters:
            vals.append(self.value(ikey))
        return vals

    def limits_of_free_parameters(self):
        """
        return the limits for each free parameter
        Return
        ------
        list
            the list of values should be in the same order as free_parameters
        """
        out = []
        for ikey in self.free_parameters:
            out.append(self.limits(ikey))
        return out

    def tag_of_free_parameters(self):
        """
        return the tag for each free parameter
        Return
        ------
        list
            the list of tag should be in the same order as free_parameters
        """
        out = []
        for ikey in self.free_parameters:
            out.append(self.tag(ikey))
        return out
