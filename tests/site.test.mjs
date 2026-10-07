import test from 'node:test';
import assert from 'node:assert/strict';
import { searchPages, confusionCounts } from '../assets/site.mjs';

test('search matches every term, ignores case, and bounds results', () => {
  const pages = Array.from({ length: 12 }, (_, i) => ({ title: `Protein ${i}`, text: 'masked residues', path: `${i}.html` }));
  assert.equal(searchPages(pages, 'PROTEIN masked').length, 8);
  assert.equal(searchPages(pages, 'unknown').length, 0);
  assert.equal(searchPages(pages, '  ').length, 0);
});

test('threshold ties are positive and counts conserve records', () => {
  const rows = [{ malignant: 0, model: .5 }, { malignant: 1, model: .5 }, { malignant: 0, model: .1 }, { malignant: 1, model: .1 }];
  assert.deepEqual(confusionCounts(rows, 'model', .5), { tn: 1, fp: 1, fn: 1, tp: 1 });
  assert.throws(() => confusionCounts(rows, 'model', NaN));
  assert.throws(() => confusionCounts(rows, 'missing', .5));
});
