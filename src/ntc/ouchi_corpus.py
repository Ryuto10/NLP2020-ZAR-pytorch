import re

import numpy as np

PRED = 'type="pred"'
GA = 'ga'
WO = 'o'
NI = 'ni'

TARGET_CASES = [GA, WO, NI]

DEP = 'dep'
ZERO = 'zero'
EXO = ['exog', 'exo1', 'exo2']

ID = 'id="(.*?)"'
GA_ID = 'ga="(.*?)"'
WO_ID = 'o="(.*?)"'
NI_ID = 'ni="(.*?)"'

GA_TYPE = 'ga_type="(.*?)"'
WO_TYPE = 'o_type="(.*?)"'
NI_TYPE = 'ni_type="(.*?)"'


def load_ntc(path):
    doc = Document(path)
    with open(path) as f:
        index = 0
        sent = Sentence(index)
        bunsetsu = None

        for line in f:
            line = line.rstrip()

            # End of sentence
            if line.startswith('EOS'):
                sent.add_bunsetsu(bunsetsu)
                doc.add_sent(sent)
                index += 1
                sent = Sentence(index)
                bunsetsu = None

            # Bunsetsu begins
            elif line.startswith('*'):
                if bunsetsu:
                    sent.add_bunsetsu(bunsetsu)
                line = line.split()
                bunsetsu = Bunsetsu(index=int(line[1]),
                                    head=int(line[2][:-1]))

            # Morphological (Token) unit
            else:
                line = line.split('\t')
                bunsetsu.add_morph(Morph(line))

    return doc


def load_file_names(path):
    file_names = []
    with open(path) as f:
        for line in f:
            line = line.rstrip()
            file_names.append(line)
    return file_names


class Morph(object):
    def __init__(self, line):
        self.id = -1
        self.index = -1
        self.bunsetsu_index = -1
        self.surface_form = line[0]

        # 追加
        if line[1].split(',')[4] != '*':
            self.base_form = line[1].split(',')[4]
        else:
            self.base_form = line[0]

        self.pos = '-'.join(line[1].split(',')[0:3])

        self.prd_arg_info = line[-1] if len(line) == 4 else ''

        self.is_predicate = False
        self.case_dict = None
        self.intra_case_dict = None

        self._set_id()
        self._set_is_predicate()
        self._set_case_dict()

    def _set_id(self):
        entity_id = re.findall(pattern=ID,
                               string=self.prd_arg_info)
        if entity_id:
            self.id = int(entity_id[0])

    def _set_is_predicate(self):
        prd_type = re.findall(pattern=PRED,
                              string=self.prd_arg_info)
        if len(prd_type) > 0:
            self.is_predicate = True
        if len(prd_type) > 1:
            print('Double predicates: %s\t%s' % (self.surface_form,
                                                 self.prd_arg_info))
            self.is_predicate = False

    def _set_case_dict(self):
        ga_id, ga_type = self._find_case_info(GA_ID, GA_TYPE)
        wo_id, wo_type = self._find_case_info(WO_ID, WO_TYPE)
        ni_id, ni_type = self._find_case_info(NI_ID, NI_TYPE)
        self.case_dict = {GA: (ga_id, ga_type),
                          WO: (wo_id, wo_type),
                          NI: (ni_id, ni_type)}

    def _find_case_info(self, case_name_ptn, case_type_ptn):
        case_id = re.findall(pattern=case_name_ptn,
                             string=self.prd_arg_info)
        case_type = re.findall(pattern=case_type_ptn,
                               string=self.prd_arg_info)

        if case_id:
            if case_id[0] in EXO:
                case_id = -1
            else:
                case_id = int(case_id[0])
        else:
            case_id = None

        if case_type:
            case_type = case_type[0]
        else:
            case_type = None

        return case_id, case_type

    def modify_case_type(self, bunsetsus):
        new_case_dict = {}
        self_bunsetsu = bunsetsus[self.bunsetsu_index]
        for case_name, (case_id, case_type) in self.case_dict.items():
            is_dep = False
            is_zero = False
            for bunsetsu in bunsetsus:
                for morph in bunsetsu.morphs:
                    if case_id == morph.id:
                        if self_bunsetsu.head == bunsetsu.index:
                            is_dep = True
                        elif self_bunsetsu.index == bunsetsu.head:
                            is_dep = True
                        else:
                            is_zero = True

            # dep is prioritized
            if is_dep:
                case_type = DEP
            elif is_zero:
                case_type = ZERO

            new_case_dict[case_name] = (case_id, case_type)

        self.case_dict = new_case_dict

    def set_intra_case_dict(self,
                            intra_ids: [int, ...],
                            same_bunsetsu_ids: [int, ...]):
        intra_case_dict = {}
        intra_ids = list(set(intra_ids) - set(same_bunsetsu_ids))
        for case_name, (case_id, case_type) in self.case_dict.items():
            if case_id in intra_ids:
                intra_case_dict[case_name] = (case_id, case_type)
            else:
                intra_case_dict[case_name] = (None, None)
        self.intra_case_dict = intra_case_dict


class Bunsetsu(object):
    def __init__(self, index, head):
        self.index = index
        self.head = head
        self.morphs = []

    def add_morph(self, morph):
        self.morphs.append(morph)


class Sentence(object):
    def __init__(self, index):
        self.index = index
        self.bunsetsus: [Bunsetsu, ...] = []
        self.morphs: [Morph, ...] = []
        self.prds: [Morph, ...] = []
        self.intra_ids: [int, ...] = []
        self.same_bunsetsu_ids: [[int, ...], ...] = []

        self.n_bunsetsus = 0
        self.n_morphs = 0
        self.n_cases = np.zeros(shape=(3, 2), dtype='int32')

    def add_bunsetsu(self, bunsetsu: Bunsetsu):
        self.bunsetsus.append(bunsetsu)
        self.n_bunsetsus += 1
        self.morphs += bunsetsu.morphs
        same_bunsetsu_ids = []

        for morph in bunsetsu.morphs:
            morph.index = self.n_morphs
            morph.bunsetsu_index = bunsetsu.index
            self.intra_ids.append(morph.id)
            same_bunsetsu_ids.append(morph.id)
            self.n_morphs += 1

            if morph.is_predicate:
                self.prds.append(morph)
        self.same_bunsetsu_ids.append(same_bunsetsu_ids)

    def set_intra_case_dict(self):
        for morph in self.prds:
            morph.modify_case_type(self.bunsetsus)
            morph.set_intra_case_dict(self.intra_ids,
                                      self.same_bunsetsu_ids[morph.bunsetsu_index])

    def count_cases(self):
        for morph in self.prds:
            for case_name in TARGET_CASES:
                case_id, case_type = morph.intra_case_dict[case_name]
                if case_id:
                    case_name_index = TARGET_CASES.index(case_name)
                    if case_type == DEP:
                        case_type_index = 0
                    else:
                        case_type_index = 1
                    self.n_cases[case_name_index][case_type_index] += 1


class Document(object):
    def __init__(self, fn):
        self.sents = []
        self.fn = '/'.join(fn.split('/')[-2:])

    def add_sent(self, sent: Sentence):
        self.sents.append(sent)


def print_stats(corpus: [Document, ...]):
    n_docs = len(corpus)
    n_sents = 0
    n_prds = 0
    n_cases = np.zeros(shape=(3, 2), dtype='int32')
    for doc in corpus:
        n_sents += len(doc.sents)
        for sent in doc.sents:
            sent.set_intra_case_dict()
            sent.count_cases()
            n_prds += len(sent.prds)
            n_cases += sent.n_cases

    print('Docs: %d' % n_docs)
    print('Sents: %d' % n_sents)
    print('Predicates: %d' % n_prds)
    print('  - GA Dep: %d  Zero: %d  TOTAL: %d' % (n_cases[0][0], n_cases[0][1], sum(n_cases[0])))
    print('  - WO Dep: %d  Zero: %d  TOTAL: %d' % (n_cases[1][0], n_cases[1][1], sum(n_cases[1])))
    print('  - NI Dep: %d  Zero: %d  TOTAL: %d' % (n_cases[2][0], n_cases[2][1], sum(n_cases[2])))
