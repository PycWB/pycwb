import os 


#It may be needed to store some other arguments where action = 'store_true' is used. 

def init_parser(parser):

     #Parse the command line arguments 
    parser.add_argument("folder", 
                        type=str, 
                        help="Folder containing the analysis")

    
    parser.add_argument('--ifo',
                        action='append', 
                        help='Call IFO to perform analysis', 
                        required=True)

    parser.add_argument("--reference", 
                        type=str, 
                        default=None, 
                        help="Name of the folder to use as absolute reference")


    parser.add_argument("--plot_median", 
                        type=bool, 
                        default=True, 
                        help="Whether to plot the median waveform")
    
    
    parser.add_argument("--confidence_level", 
                        type=float, 
                        default=0.9, 
                        help="Confidence level for the analysis")


    parser.add_argument("--waveform_format", 
                        type=str, 
                        default="hdf", 
                        help="Format of the waveform files. Options are 'hdf' or 'txt'")


    parser.add_argument("--ordering", 
                        type=str, 
                        default="percentiles", 
                        help="Method to compute the confidence level. Options are 'percentile', 'lower', 'upper'")

    parser.add_argument("--whitened", 
                        action="store_true", 
                        help="Whether to use whitened waveforms")



    
    parser.add_argument("--max_workers",
                         type=int, 
                         default=8, 
                         help="Maximum number of workers for parallel processing")
    


def command(args):
    import sys
    sys.path.insert(0, '/home/alessandro.martini/pycwb/pycwb/workflow/subflow/')
    from waveform_reconstruction_report import process_strain
    #from pycwb.workflow.subflow.waveform_reconstruction_report import process_strain 
    
    for ifo in args.ifo:
        #check process_strain argumenta are correctly defined 
        process_strain(folder= args.folder, 
                    ifo = ifo, 
                    reference = args.reference, 
                    confidence_level = args.confidence_level, 
                    whitened = args.whitened,
                    waveform_format = args.waveform_format, 
                    ordering = args.ordering, 
                    plot_median = args.plot_median, 
                    max_workers = args.max_workers)
