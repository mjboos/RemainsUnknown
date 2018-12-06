from allennlp.common.testing import AllenNlpTestCase
from allennlp.common.util import ensure_list

from remainsunknown.dataset_readers import SemanticScholarRemainsReader

class TestSemanticScholarRemainsReader(AllenNlpTestCase):
    def test_read_from_file(self):
        reader = SemanticScholarRemainsReader()
        instances = ensure_list(reader.read('tests/fixtures/s2_papers.jsonl'))
        instance1 = {"title" : ['The', 'anterior', 'gradient', 'homolog', '3', '(',
                                'AGR3', ')', 'gene', 'is', 'associated', 'with',
                                'differentiation', 'and', 'survival', 'in', 'ovarian',
                                'cancer', '.'],
                      "paperAbstract" : ['Low', '-', 'grade', '(', 'LG', ')', 'serous',
                                         'ovarian', 'carcinoma', 'is'],
                       "remains" : ["UNAPPLICABLE"]}
        instance2 = {'paperAbstract' : ['Antibodies', 'against', 'CD66', 'identify',
        'antigens', 'from', 'the', 'carcinoembryonic', 'antigen', '('],
                      'title' : ['CD66', 'expression', 'in', 'acute', 'leukaemia'],
                      'remains' : ['The', 'association', 'between', 'CD66',
                       'reactivity', 'and', 'bcr', '-', 'abl', 'in', 'adult',
                       'ALL', 'remains', 'to', 'be', 'investigated', '.']}
        assert len(instances) == 10
        fields = instances[0].fields
        assert [t.text for t in fields["title"].tokens] == instance1["title"]
        assert [t.text for t in fields["abstract"].tokens[:len(instance1["paperAbstract"])]] == instance1["paperAbstract"]
        assert fields['remains'].tokens[0].text == instance1['remains'][0]
        fields = instances[2].fields
        assert [t.text for t in fields["title"].tokens] == instance2["title"]
        assert [t.text for t in fields["abstract"].tokens[:len(instance2["paperAbstract"])]] == instance2["paperAbstract"]
        assert [t.text for t in fields['remains'].tokens] == instance2['remains']