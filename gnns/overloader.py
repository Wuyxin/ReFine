from functools import wraps

def overload(func):
    @wraps(func)
    def wrapper(*args, **kargs):
        if len(args) +  len(kargs) == 2:
            if len(args) == 2: # for inputs like model(g)
                g = args[1]
            else:# for inputs like model(graph=g)
                g = kargs['graph']
            return func(args[0], g.x, g.edge_index, g.edge_attr, g.batch)

        elif len(args) +  len(kargs) == 5:
            if len(args) == 5: # for inputs like model(x, ..., batch)
                return func(*args)
            else: # for inputs like model(x=x, ..., batch=batch)
                return func(args[0], **kargs)

        elif len(args) +  len(kargs) == 6:
            if len(args) == 6: # for inputs like model(x, ..., batch, pos)
                return func(*args[:-1])
            else: # for inputs like model(x=x, ..., batch=batch, pos=pos)
                return func(args[0], kargs['x'], kargs['edge_index'], kargs['edge_attr'], kargs['batch'])
        else:
            raise TypeError
    return wrapper