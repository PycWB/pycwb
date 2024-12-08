import matplotlib.pyplot as plt

def report(ifar_rho_source=None, **kwargs):

    print(f"Reporting the results")
    if ifar_rho_source:
        if ifar_rho_source not in kwargs:
            print(f"Source {ifar_rho_source} not found in the results")
            raise ValueError(f"Source {ifar_rho_source} not found in the results")
        data = kwargs[ifar_rho_source]

        # plot the ifar vs ranking parameter
        plt.plot(data['bins'], data['ifar'], drawstyle='steps-post')
        plt.xlabel(data['ranking_par'])
        plt.ylabel('ifar')
        plt.yscale('log')
        plt.savefig(f"{ifar_rho_source}.png")
        plt.close()

        # plot the number of events vs ranking parameter
        plt.plot(data['bins'], data['n_events'], drawstyle='steps-post')
        plt.xlabel(data['ranking_par'])
        plt.ylabel('number of events')
        plt.yscale('log')
        plt.savefig(f"{ifar_rho_source}_n_events.png")
        plt.close()