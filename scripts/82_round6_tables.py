#!/usr/bin/env python
"""Render generated-neutral training and independent calibration without causal claims."""
from pathlib import Path
import json

ROOT=Path(__file__).resolve().parents[1]
LABEL={'olmo2-1b-it':'OLMo-2-1B-I','llama32-3b':'Llama-3.2-3B'}


def load():
    path=ROOT/'results/summary/round6_generated.json'
    if not path.exists():return None
    data=json.loads(path.read_text())
    if not data.get('validated'):raise ValueError('unvalidated generated-neutral follow-up')
    return data


def ci(value):
    return '--' if value is None else f'{100*value[0]:.1f} [{100*value[1]:.1f}, {100*value[2]:.1f}]'


def table(lines,columns,header,rows,caption,label):
    lines.extend([r'\begin{table*}[t]',r'\centering\small',r'\sbox{\roundsixbox}{\begin{tabular}{@{}'+columns+r'@{}}',
                  r'\toprule',header+r' \\',r'\midrule'])
    lines.extend(' & '.join(row)+r' \\' for row in rows)
    lines.extend([r'\bottomrule',r'\end{tabular}}',
                  r'\ifdim\wd\roundsixbox>\textwidth\resizebox{\textwidth}{!}{\usebox{\roundsixbox}}\else\usebox{\roundsixbox}\fi',
                  r'\caption{'+caption+'}',r'\label{'+label+'}',r'\end{table*}'])


def narrative_numbers():
    data=load();out={}
    if data:
        for model,tag in (('olmo2-1b-it','olmo1'),('llama32-3b','llama3')):
            cells=[cell for role in data['models'][model].values() for cell in role['cells'].values()]
            rates=[c['neutral_calibration_ci95'][0] for c in cells]
            out[f'r6-generated-{tag}-cal-min']=f'{100*min(rates):.1f}'
            out[f'r6-generated-{tag}-cal-max']=f'{100*max(rates):.1f}'
            out[f'r6-generated-{tag}-cal-passed']=str(sum(c['calibration_pass'] for c in cells))
            out[f'r6-generated-{tag}-cal-cells']=str(len(cells))
        if 'source_cohorts' in data:
            for key,value in data['source_cohorts'].items():out[f'r6-source-{key}']=f'{value:,}'
    return out


def main():
    data=load();lines=['% Generated from validated round-six summary.',r'\newsavebox{\roundsixbox}']
    if data:
        lines.extend([r'\section{Training probes on generated neutral chains}',r'\label{app:round6-generated}', r'\input{figures/readout_check}',
            'Gold-trained probe transfer (Appendix~\\ref{app:round5-generated}) may fail because the states along generated chains differ from those along the supplied correct chain. We therefore train on freely generated neutral chains from \\NUM{r6-source-train_n} source problems and select layers on \\NUM{r6-source-validation_n} separate validation problems. Canonical equations partition the source pool, keeping renamed versions of a computation together. The training and validation pools contain \\NUM{r6-source-train_keys} and \\NUM{r6-source-validation_keys} computations; both exclude all test and demonstration computations. These counts distinguish independent computations from repeated name renderings.',
            'Generation is greedy with demonstration set 7 and a 128-token limit. The probe sees the hidden state before the target variable\'s first numeric assignment, with every token boundary checked to exclude the value digit. Missing or unsafe boundaries are excluded and counted. We retain all usable neutral chains regardless of whether the model writes the correct value, using the executed true value as the training label. Three probe seeds use the original full-batch SGD recipe (learning rate 0.001, 10,000 epochs). States are stored in float32 and nonfinite states or fitted weights are rejected. Layers are searched at stride two, including the final layer, and chosen only by mean validation accuracy with the lowest-layer tie break. The name-identity control is trained separately and does not select layers.',
            'Table~\\ref{tab:round6-selection} reports the usable source counts and selected layers. Evaluation uses exactly the earlier fixed 200-set sample and separately reported additional name-error cases under demonstration sets 7, 11 and 13. Table~\\ref{tab:round6-calibration} gives held-out neutral calibration and boundary coverage. Intervals use 4,000 canonical-computation bootstrap draws, keeping repeated names together. The 90\\% calibration threshold applies to the point estimate, not the lower interval endpoint.',
            'Llama passes all six calibration cells; OLMo passes none. Retraining on generated chains therefore does not validate a strong readout account of OLMo\'s own errors. Table~\\ref{tab:round6-errors} retains its error readouts descriptively. Llama has only one name-error case in the primary sample and thirteen additional observed writes; these rare, selected cases cannot establish a general mechanism. Paired changes from gold-trained transfer are shown in Table~\\ref{tab:round6-changes}; they compare the same eligible prefixes, not independent model samples.'])
        selections=[];calibration=[];errors=[];changes=[]
        for model,roles in data['models'].items():
            for role,result in roles.items():
                selection=result['selection'];chosen=next(r for r in selection['validation_scores'] if r['layer']==selection['layer'])
                selections.append([LABEL[model],role,str(selection['n_train_valid']),str(selection['n_validation_valid']),
                                   str(selection['layer']),f"{100*chosen['accuracy']:.1f}",f"{100*chosen['control_accuracy']:.1f}"])
                for seed,cell in result['cells'].items():
                    rows=[r for r in data['records'] if r['model']==model and r['role']==role and str(r['seed'])==seed
                          and r['condition']=='neutral' and r['primary_sample']]
                    coverage=f"{sum(r['boundary_valid'] for r in rows)}/{len(rows)}"
                    calibration.append([LABEL[model],role,seed,coverage,ci(cell['neutral_calibration_ci95']),
                                        ci(cell['neutral_control_ci95']),'yes' if cell['calibration_pass'] else 'no'])
                    primary=cell['lure_writes']['primary_sample'];extra=cell['lure_writes']['lure_augmentation']
                    errors.append([LABEL[model],role,seed,str(primary['n']),ci(primary['accuracy_ci95']),str(extra['n']),ci(extra['accuracy_ci95'])])
                    changes.append([LABEL[model],role,seed,ci(cell['neutral_change_from_gold_ci95']),
                                    ci(primary['change_from_gold_ci95']),ci(extra['change_from_gold_ci95'])])
        table(lines,'llrrrrr','Model & Role & Train $n$ & Validation $n$ & Layer & Validation (\\%) & Name ID (\\%)',selections,
              'Probe source rows with usable generated boundaries, selected layers and validation scores. Source cohorts are computation-disjoint; usable counts vary by model and variable. Validation scores select layers and are not test estimates.','tab:round6-selection')
        table(lines,'llrlllc','Model & Role & Set & Neutral coverage & Calibration (\\%) & Name ID (\\%) & Pass',calibration,
              'Held-out primary neutral generated chains. Coverage counts valid pre-value boundaries over all primary neutral rows. Accuracy intervals are 95\\% canonical-computation bootstrap intervals; pass means the calibration point estimate meets 90\\%.','tab:round6-calibration')
        table(lines,'llrrlrl','Model & Role & Set & Sample $n$ & Sample accuracy & Extra $n$ & Extra accuracy',errors,
              'True-digit accuracy (\\%) [95\\% interval] on actual name errors, with the fixed sample and additional error-selected cases separated. Extra cases exclude the primary sample. Dashes denote empty strata. OLMo fails neutral calibration, so its readouts are descriptive.','tab:round6-errors')
        table(lines,'llrlll','Model & Role & Set & Neutral change & Sample error change & Extra error change',changes,
              'Paired accuracy changes in percentage points [95\\% interval] from gold-trained transfer to generated-neutral training on the same valid prefixes. Error-selection and failed neutral calibration limit the interpretation.','tab:round6-changes')
    (ROOT/'paper/tables/round6_followups.tex').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main()
