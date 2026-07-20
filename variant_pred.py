import jax
from alphagenome.data import genome
from alphagenome.visualization import plot_components
from alphagenome_research.model import dna_model
import matplotlib.pyplot as plt

#model = dna_model.create_from_kaggle('all_folds')
model = dna_model.create(
    checkpoint_path = '/scratch/03762/eriksf/alphagenome_research/models/all_folds/',
    organism_settings = {
        dna_model.Organism.HOMO_SAPIENS: dna_model.OrganismSettings(
            fasta_path='./reference/gencode/hg38/GRCh38.p13.genome.fa',
            gtf_feather_path='./reference/gencode/hg38/gencode.v46.annotation.gtf.gz.feather',
            pas_feather_path='./reference/exon/hg38/polyadb_human_v3_exon3_contiguous_gtfv46.feather',
            splice_site_starts_feather_path='./reference/gencode/hg38/gencode.v46.splice_sites_starts.feather',
            splice_site_ends_feather_path='./reference/gencode/hg38/gencode.v46.splice_sites_ends.feather',
            calibration_path='./reference/hg38/calibration_scores.pb'
        ),
        dna_model.Organism.MUS_MUSCULUS: dna_model.OrganismSettings(
            fasta_path='./reference/gencode/mm10/GRCm38.p6.genome.fa',
            gtf_feather_path='./reference/gencode/mm10/gencode.vM23.annotation.gtf.gz.feather',
            pas_feather_path=None,
            splice_site_starts_feather_path='./reference/gencode/mm10/gencode.vM23.splice_sites_starts.feather',
            splice_site_ends_feather_path='./reference/gencode/mm10/gencode.vM23.splice_sites_ends.feather',
            calibration_path=None
        )
    },
    device=jax.local_devices()[0]
)

interval = genome.Interval(chromosome='chr22', start=35677410, end=36725986)
variant = genome.Variant(
    chromosome='chr22',
    position=36201698,
    reference_bases='A',
    alternate_bases='C',
)

outputs = model.predict_variant(
    interval=interval,
    variant=variant,
    ontology_terms=['UBERON:0001157'],
    requested_outputs=[dna_model.OutputType.RNA_SEQ],
)

plot_components.plot(
    [
        plot_components.OverlaidTracks(
            tdata={
                'REF': outputs.reference.rna_seq,
                'ALT': outputs.alternate.rna_seq,
            },
            colors={'REF': 'dimgrey', 'ALT': 'red'},
        ),
    ],
    interval=outputs.reference.rna_seq.interval.resize(2**15),
    # Annotate the location of the variant as a vertical line.
    annotations=[plot_components.VariantAnnotation([variant], alpha=0.8)],
)
#plt.show()
plt.savefig('pv.png')
