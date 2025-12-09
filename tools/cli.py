import click
from pathlib import Path
from scenarios import SCENARIO_PRESETS, load_and_apply_scenario
from visualization import run_visual_simulation
from main import _run_single_trial, run_batch  # reuse existing functions


@click.group()
def cli():
    pass


@cli.command()
def list_scenarios():
    for k in sorted(SCENARIO_PRESETS.keys()):
        click.echo(k)


@cli.command()
@click.argument("scenario")
def visual(scenario):
    env, meta = load_and_apply_scenario(scenario)
    run_visual_simulation(env)


@cli.command()
@click.argument("scenario")
@click.option("--agents", type=int)
@click.option("--steps", type=int)
@click.option("--target-percent", type=float)
@click.option("--out-dir", type=click.Path())
def run(scenario, agents, steps, target_percent, out_dir):
    res = _run_single_trial(
        scenario,
        agents,
        steps,
        target_percent,
        trial_index=1,
        out_dir=Path(out_dir or "."),
        overlay=False,
    )
    click.echo(res)


@cli.command()
@click.argument("scenario")
@click.option("--trials", default=5, type=int)
@click.option("--workers", default=2, type=int)
def batch(scenario, trials, workers):
    run_batch(scenario, trials=trials, workers=workers)


if __name__ == "__main__":
    cli()
