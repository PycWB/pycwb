"""CLI entry point for pycwb post-production workflow."""

import os
import webbrowser


def init_parser(parser):
    # Add the arguments
    parser.add_argument('workflow_file',
                        metavar='file_path',
                        type=str,
                        help='the path to the workflow file')
    parser.add_argument('--diagram-only', '-d',
                        action='store_true',
                        help='Generate workflow DAG diagram and exit (dry-run, '
                             'no execution).')
    parser.add_argument('--open', '-o',
                        action='store_true',
                        help='Open the interactive HTML diagram in browser '
                             'after generation (only with --diagram-only).')
    parser.add_argument('--no-diagram',
                        action='store_true',
                        help='Skip diagram generation during workflow execution.')


def command(args):
    from pycwb.post_production.workflow import run_workflow
    from pycwb.post_production.diagram import generate_workflow_diagram

    if args.diagram_only:
        result = generate_workflow_diagram(args.workflow_file)
        print(f"Diagram generated:")
        for fmt, path in sorted(result.items()):
            if path and fmt not in ('dag', 'png_method'):
                print(f"  {fmt}: {path}")
        print(f"  nodes: {len(result['dag']['nodes'])}, "
              f"edges: {len(result['dag']['edges'])}")

        # Open interactive HTML in browser (opt-in only)
        html_path = result.get('html')
        if html_path and args.open:
            abs_path = os.path.abspath(html_path)
            webbrowser.open(f'file://{abs_path}')
            print(f"\n  Opened {abs_path} in browser")
        return

    run_workflow(
        args.workflow_file,
        generate_diagram=not args.no_diagram,
    )