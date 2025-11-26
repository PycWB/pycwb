import os 


#It may be needed to store some other arguments where action = 'store_true' is used. 

def init_parser(parser):

     #Parse the command line arguments 
    parser.add_argument("folder", 
                        type=str, 
                        help="Folder containing the analysis")

    
   # parser.add_argument("--ifos", 
   #                     type=list, 
   #                     default=["H1","L1"],
   #                     help="List of ifos used in the analysis")
    
    parser.add_argument('--ifo',
                        action='append', 
                        help='Call IFO to perform analysis', 
                        required=True)



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


    parser.add_argument("--use_absolute_reference", 
                        action="store_true", 
                        help="Whether to use absolute reference")

        
    parser.set_defaults(use_absolute_reference=True)


    
    parser.add_argument("-no_absolute_reference", dest="use_absolute_reference", action="store_false", help="Whether to not use absolute reference")



    parser.add_argument("--reference_folder", 
                        type=str, 
                        default=None, 
                        help="Name of the folder to use as absolute reference")
    
    parser.add_argument("--use_relative_reference", 
                        action="store_true", 
                        help="Whether to use relative reference")
    


def command(args):
    from pycwb.workflow.subflow.waveform_reconstruction_report import process_strain 

    if not args.use_absolute_reference and not args.use_relative_reference:
        raise ValueError("At least one of --use_absolute_reference or --use_relative_reference must be set to True.") 
    
    for ifo in args.ifo:
        #check process_strain argumenta are correctly defined 
        process_strain(folder= args.folder, 
                    ifo = ifo, 
                    reference_folder= args.reference_folder, 
                    confidence_level = args.confidence_level, 
                    use_absolute_reference = args.use_absolute_reference, 
                    use_relative_reference = args.use_relative_reference, 
                    waveform_format = args.waveform_format, 
                    ordering = args.ordering, 
                    plot_median = args.plot_median)