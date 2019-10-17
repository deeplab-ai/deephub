from unittest import mock

import click.testing

from deephub.trainer.__main__ import cli


class TestTrainerCLI:

    def _invoke_cli(self, *args):
        runner = click.testing.CliRunner()
        return runner.invoke(cli, args=args)

    def test_train_unknown(self):
        result = self._invoke_cli('train', 'unknown')
        assert result.exit_code != 0
        assert "Error: Cannot find variant with name 'unknown'" in result.output

    def test_train_packaged_variant(self, datadir):
        variants_dir = datadir / 'config' / 'variants'
        with mock.patch('deephub.trainer.__main__.load_variant') as mocked_load_variant, \
                mock.patch('deephub.trainer.__main__.Trainer') as mocked_trainer:
            mocked_pd = mock.MagicMock()
            mocked_load_variant.return_value = mocked_pd

            result = self._invoke_cli('train', 'varianta', '-d', variants_dir)

            # Check that objects instantiated properly
            mocked_load_variant.assert_called_once_with(variant_name='varianta', variants_dir=variants_dir)
            mocked_trainer.assert_called_once_with(requested_gpu_devices=())

            # Check that was training was called correctly
            mocked_pd.train.assert_called_once()
            assert result.exit_code == 0
            assert 'finished' in result.output

    def test_train_packaged_with_user_definitions(self, datadir):
        variants_dir = datadir / 'config' / 'variants'
        with mock.patch('deephub.variants.io.VariantDefinition') as mocked_pd, \
                mock.patch('deephub.trainer.__main__.Trainer') as mocked_trainer:
            # Test unknown
            result = self._invoke_cli('train', 'unknown', '-d', variants_dir)
            assert result.exit_code != 0
            assert "Error: Cannot find variant with name 'unknown'" in result.output

        with mock.patch('deephub.variants.io.VariantDefinition') as mocked_pd, \
                mock.patch('deephub.trainer.__main__.Trainer') as mocked_trainer:
            result = self._invoke_cli('train', 'varianta', '-d', variants_dir)

            mocked_pd.assert_called_once()
            assert mocked_pd.mock_calls[0][1][0] == 'varianta'

            assert result.exit_code == 0
            assert 'finished' in result.output

    def test_train_packaged_with_multiple_user_definitions(self, datadir):
        variants_dir = datadir / 'config' / 'variants'
        with mock.patch('deephub.variants.io.VariantDefinition') as mocked_pd, \
                mock.patch('deephub.trainer.__main__.Trainer') as mocked_trainer:
            # Test unknown
            result = self._invoke_cli('train', 'unknown', '-d', variants_dir)
            assert result.exit_code != 0
            assert "Error: Cannot find variant with name 'unknown'" in result.output

        with mock.patch('deephub.variants.io.VariantDefinition') as mocked_pd, \
                mock.patch('deephub.trainer.__main__.Trainer') as mocked_trainer:
            # Test unknown
            result = self._invoke_cli('train', 'variantc', '-d', variants_dir)

            mocked_pd.assert_called_once()
            assert mocked_pd.mock_calls[0][1][0] == 'variantc'

            assert result.exit_code == 0
            assert 'finished' in result.output

    def test_override_configuration_with_literals(self, datadir):
        variants_dir = datadir / 'config' / 'variants'
        # Check that VariantDefinition.set was used
        with mock.patch('deephub.variants.io.VariantDefinition') as mocked_pd, \
                mock.patch('deephub.trainer.__main__.Trainer') as mocked_trainer:
            # Test unknown
            result = self._invoke_cli('train', 'varianta',
                                      '-d', variants_dir,
                                      '-p', 'model.variable', '15',
                                      '-p', 'yet.another', 'mitsos')

            mocked_pd.assert_called_once()
            assert len(mocked_pd().set.mock_calls) == 2
            mocked_pd().set.assert_has_calls([mock.call('model.variable', 15),
                                              mock.call('yet.another', 'mitsos')])
            assert result.exit_code == 0
            assert 'finished' in result.output

    def test_override_configuration_with_expr(self, datadir):
        variants_dir = datadir / 'config' / 'variants'
        # Check that VariantDefinition.set was used
        with mock.patch('deephub.variants.io.VariantDefinition') as mocked_pd, \
                mock.patch('deephub.trainer.__main__.Trainer') as mocked_trainer:
            # Test unknown
            result = self._invoke_cli('train', 'varianta',
                                      '-d', variants_dir,
                                      '-p', 'mixed.list', '[12, "mixed", 3.0]',
                                      '-p', 'math.expression', '12/3.0')

            mocked_pd.assert_called_once()
            assert len(mocked_pd().set.mock_calls) == 2
            mocked_pd().set.assert_has_calls([mock.call('mixed.list', [12, "mixed", 3.0]),
                                              mock.call('math.expression', 4.0)])
            assert result.exit_code == 0
            assert 'finished' in result.output

    def test_override_configuration_with_str(self, datadir):
        variants_dir = datadir / 'config' / 'variants'
        # Check that VariantDefinition.set usage
        with mock.patch('deephub.variants.io.VariantDefinition') as mocked_pd, \
                mock.patch('deephub.trainer.__main__.Trainer') as mocked_trainer:
            # Test unknown
            result = self._invoke_cli('train', 'varianta',
                                      '-d', variants_dir,
                                      '-P', 'mixed.list', '[12, "mixed", 3.0]',
                                      '-P', 'math.expression', '12/3.0')

            # mocked_pd.assert_called_once()
            # assert len(mocked_pd().set.mock_calls) == 2
            # mocked_pd().set.assert_has_calls([mock.call('mixed.list', [12, "mixed", 3.0]),
            #                                   mock.call('math.expression', 4.0)])
            assert result.exit_code != 0
            assert 'implemented' in result.output

    def test_no_gpu_configuration(self, datadir):
        variants_dir = datadir / 'config' / 'variants'
        # Check that VariantDefinition.set was used
        with mock.patch('deephub.variants.io.VariantDefinition') as mocked_pd, \
                mock.patch('deephub.trainer.__main__.Trainer') as mocked_trainer:
            result = self._invoke_cli('train', 'varianta',
                                      '-d', variants_dir)

            mocked_trainer.assert_called_once_with(requested_gpu_devices=())
            assert result.exit_code == 0

    def test_single_gpu_configuration(self, datadir):
        variants_dir = datadir / 'config' / 'variants'
        # Check that VariantDefinition.set was used
        with mock.patch('deephub.variants.io.VariantDefinition') as mocked_pd, \
                mock.patch('deephub.trainer.__main__.Trainer') as mocked_trainer:
            result = self._invoke_cli('train', 'varianta',
                                      '-d', variants_dir,
                                      '-g', '0')

            mocked_trainer.assert_called_once_with(requested_gpu_devices=(0,))
            assert result.exit_code == 0

    def test_multi_gpu_configuration(self, datadir):
        variants_dir = datadir / 'config' / 'variants'
        # Check that VariantDefinition.set was used
        with mock.patch('deephub.variants.io.VariantDefinition') as mocked_pd, \
                mock.patch('deephub.trainer.__main__.Trainer') as mocked_trainer:
            result = self._invoke_cli('train', 'varianta',
                                      '-d', variants_dir,
                                      '-g', '1', '-g', '0')

            mocked_trainer.assert_called_once_with(requested_gpu_devices=(1, 0))
            assert result.exit_code == 0

    def test_gpu_wrong_id(self, datadir):
        variants_dir = datadir / 'config' / 'variants'
        # Check that VariantDefinition.set was used
        with mock.patch('deephub.variants.io.VariantDefinition') as mocked_pd, \
                mock.patch('deephub.trainer.__main__.Trainer') as mocked_trainer:
            result = self._invoke_cli('train', 'varianta',
                                      '-d', variants_dir,
                                      '-g', 'GPU:0')

            assert result.exit_code != 0
            assert "Invalid value for " in result.output

    def test_warm_start_invocation(self, datadir, tmpdir):
        variants_dir = datadir / 'config' / 'variants'
        # Check that VariantDefinition.set was used
        with mock.patch('deephub.variants.io.VariantDefinition') as mocked_pd, \
                mock.patch('deephub.trainer.__main__.Trainer') as mocked_trainer:
            # Test unknown
            result = self._invoke_cli('train', 'varianta',
                                      '-d', variants_dir,
                                      '--warm-start-checkpoint', str(tmpdir),
                                      '--warm-start-vars', "myscope1")

            mocked_pd.assert_called_once()
            assert len(mocked_pd().set.mock_calls) == 2
            mocked_pd().set.assert_has_calls([mock.call('train.warm_start_check_point', str(tmpdir)),
                                              mock.call('train.warm_start_variables_regex', "myscope1")])
            assert result.exit_code == 0
            assert 'finished' in result.output
