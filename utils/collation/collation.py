import torch
import MinkowskiEngine as ME


class CollateFN:
    def __init__(self, device=None):
        self.device = device

    def __call__(self, list_data):
        r"""
        Collation function for MinkowskiEngine.SparseTensor that creates batched
        coordinates given a list of dictionaries.
        """
        list_d = []
        list_idx = []
        for d in list_data:
            list_d.append((d["coordinates"].to(self.device), d["features"].to(self.device), d["labels"]))
            list_idx.append(d["idx"].view(-1, 1))

        coordinates_batch, features_batch, labels_batch = ME.utils.SparseCollation(dtype=torch.float32,
                                                                                   device=self.device)(list_d)
        idx = torch.cat(list_idx, dim=0)
        return {"coordinates": coordinates_batch,
                "features": features_batch,
                "labels": labels_batch,
                "idx": idx}


class CollateFNPseudo:
    def __init__(self, device=None):
        self.device = device

    def __call__(self, list_data):
        r"""
        Collation function for MinkowskiEngine.SparseTensor that creates batched
        coordinates given a list of dictionaries.
        """
        list_data_batch = []
        list_pseudo = []

        selected = []
        idx = []

        for d in list_data:
            list_data_batch.append((d["coordinates"].to(self.device), d["features"].to(self.device), d["labels"]))
            list_pseudo.append(d["pseudo_labels"].to(self.device))
            selected.append(d["sampled_idx"].unsqueeze(0))
            idx.append(d["idx"].unsqueeze(0))

        coordinates_batch, features_batch, labels_batch = ME.utils.SparseCollation(dtype=torch.float32,
                                                                                   device=self.device)(list_data_batch)

        idx = torch.cat(idx, dim=0)
        pseudo_labels = torch.cat(list_pseudo, dim=0)

        return {"coordinates": coordinates_batch,
                "features": features_batch,
                "labels": labels_batch,
                "pseudo_labels": pseudo_labels,
                "sampled_idx": selected,
                "idx": idx}


class CollateMerged:
    def __init__(self,
                 device: torch.device = torch.device("cpu")) -> None:
        self.device = device

    def __call__(self, list_data) -> dict:
        r"""
        Collation function for MinkowskiEngine.SparseTensor that creates batched
        coordinates given a list of dictionaries.
        """

        source_list_data = [(d["source_coordinates"].to(self.device), d["source_features"].to(self.device), d["source_labels"]) for d in list_data]
        target_list_data = [(d["target_coordinates"].to(self.device), d["target_features"].to(self.device), d["target_labels"]) for d in list_data]

        source_coordinates_batch, source_features_batch, source_labels_batch = ME.utils.SparseCollation(dtype=torch.float32,
                                                                                                        device=self.device)(source_list_data)

        target_coordinates_batch, target_features_batch, target_labels_batch = ME.utils.SparseCollation(dtype=torch.float32,
                                                                                                        device=self.device)(target_list_data)

        return_dict = {"source_coordinates": source_coordinates_batch,
                       "source_features": source_features_batch,
                       "source_labels": source_labels_batch,
                       "target_coordinates": target_coordinates_batch,
                       "target_features": target_features_batch,
                       "target_labels": target_labels_batch}

        return return_dict


class CollateMergedPseudo:
    def __init__(self, device=None):
        self.device = device

    def __call__(self, list_data):
        r"""
        Collation function for MinkowskiEngine.SparseTensor that creates batched
        coordinates given a list of dictionaries.
        """
        source_list_data = []
        target_list_data = []
        target_list_pseudo = []

        source_selected = []
        target_selected = []

        source_idx = []
        target_idx = []

        for d in list_data:
            source_list_data.append((d["source_coordinates"].to(self.device), d["source_features"].to(self.device), d["source_labels"]))
            target_list_data.append((d["target_coordinates"].to(self.device), d["target_features"].to(self.device), d["target_labels"]))
            target_list_pseudo.append((d["target_coordinates"].to(self.device), d["target_features"].to(self.device), d["target_pseudo_labels"].to(self.device)))

            source_selected.append(d["source_sampled_idx"])
            target_selected.append(d["target_sampled_idx"])
            source_idx.append(d["source_idx"].unsqueeze(0))
            target_idx.append(d["target_idx"].unsqueeze(0))

        source_coordinates_batch, source_features_batch, source_labels_batch = ME.utils.SparseCollation(dtype=torch.float32,
                                                                                   device=self.device)(source_list_data)

        target_coordinates_batch, target_features_batch, target_labels_batch = ME.utils.SparseCollation(dtype=torch.float32,
                                                                                   device=self.device)(target_list_data)

        _, _, target_pseudo_labels = ME.utils.SparseCollation(dtype=torch.float32,
                                                             device=self.device)(target_list_pseudo)

        return {"source_coordinates": source_coordinates_batch,
                "source_features": source_features_batch,
                "source_labels": source_labels_batch,
                "target_coordinates": target_coordinates_batch,
                "target_features": target_features_batch,
                "target_labels": target_labels_batch,
                "target_pseudo_labels": target_pseudo_labels,
                "source_sampled": source_selected,
                "target_sampled": target_selected,
                "source_idx": source_idx,
                "target_idx": target_idx}


class CollateMixed:
    def __init__(self, device=None):
        self.device = device

    def __call__(self, list_data):
        r"""
        Collation function for MinkowskiEngine.SparseTensor that creates batched
        coordinates given a list of dictionaries.
        list_data has a list of dicts with keys:
            - mixed_coordinates
            - mixed_labels
            - mixed_gt_labels
            - mixed_features
            - separation_idx
            - mixed_sampled_idx
            - mixed_idx
        """
        gt_list_data = []
        pseudo_list_data = []

        separation_list = []
        mixed_sampled = []
        mixed_idx = []

        for d in list_data:
            gt_list_data.append((d["mixed_coordinates"].to(self.device), d["mixed_features"].to(self.device), d["mixed_gt_labels"]))
            pseudo_list_data.append((d["mixed_coordinates"].to(self.device), d["mixed_features"].to(self.device), d["mixed_labels"]))

            mixed_sampled.append(d["mixed_sampled"])
            separation_list.append(d["separation_idx"].unsqueeze(0))
            mixed_idx.append(d["mixed_idx"].unsqueeze(0))

        _, _, gt_labels_batch = ME.utils.SparseCollation(dtype=torch.float32,
                                                         device=self.device)(gt_list_data)

        pseudo_coordinates_batch, pseudo_features_batch, pseudo_labels_batch = ME.utils.SparseCollation(dtype=torch.float32,
                                                                                   device=self.device)(pseudo_list_data)

        separation_list = torch.cat(separation_list, dim=0)
        mixed_idx = torch.cat(mixed_idx, dim=0)

        return {"mixed_coordinates": pseudo_coordinates_batch,
                "mixed_features": pseudo_features_batch,
                "mixed_labels": pseudo_labels_batch,
                "mixed_gt_labels": gt_labels_batch,
                "mixed_sampled": mixed_sampled,
                "separation_idx": separation_list,
                "mixed_idx": mixed_idx}


class CollateMixedMasked:
    def __init__(self, device=None):
        self.device = device

    def __call__(self, list_data):
        r"""
        Collation function for MinkowskiEngine.SparseTensor that creates batched
        coordinates given a list of dictionaries.
        list_data has a list of dicts with keys:

        """
        source_list = []
        target_list = []
        masked_list = []
        source_idx = []
        target_idx = []
        masked_idx = []
        masked_mask = []
        source_masked_mask = []
        target_masked_mask = []

        source_masked_idx = []
        target_masked_idx = []
        source_common_index = []
        target_common_index = []

        for d in list_data:
            source_list.append((d["source_coordinates"].to(self.device), d["source_features"].to(self.device), d["source_labels"]))
            target_list.append((d["target_coordinates"].to(self.device), d["target_features"].to(self.device), d["target_labels"]))
            masked_list.append((d["s2t_masked_coordinates"].to(self.device), d["s2t_masked_features"].to(self.device), d["s2t_masked_labels"]))

            source_idx.append(d["source_idx"].unsqueeze(0))
            target_idx.append(d["target_idx"].unsqueeze(0))
            masked_idx.append(d["s2t_masked_idx"])

            # masked_mask.append(d["s2t_masked_mask"])

            source_masked_mask.append(d["s2t_masked_source_mask"])
            target_masked_mask.append(d["s2t_masked_target_mask"])

            source_common_index.append(d['source_common_index'])
            target_common_index.append(d['target_common_index'])

        source_coordinates_batch, source_features_batch, source_labels_batch = ME.utils.SparseCollation(dtype=torch.float32,
                                                                                                        device=self.device)(source_list)

        target_coordinates_batch, target_features_batch, target_labels_batch = ME.utils.SparseCollation(dtype=torch.float32,
                                                                                                        device=self.device)(target_list)

        masked_coordinates_batch, masked_features_batch, masked_labels_batch = ME.utils.SparseCollation(dtype=torch.float32,
                                                                                                        device=self.device)(masked_list)

        source_idx = torch.cat(source_idx, dim=0)
        target_idx = torch.cat(target_idx, dim=0)
        masked_idx = torch.cat(masked_idx, dim=0)

        # masked_mask = torch.cat(masked_mask, dim=0)
        source_masked_mask = torch.cat(source_masked_mask, dim=0)
        target_masked_mask = torch.cat(target_masked_mask, dim=0)
        source_common_index = torch.cat(source_common_index, dim=0)
        target_common_index = torch.cat(target_common_index, dim=0)

        return {"source_coordinates": source_coordinates_batch,
                "source_labels": source_labels_batch,
                "source_features": source_features_batch,
                "source_common_index": source_common_index,
                "source_idx": source_idx,
                "target_coordinates": target_coordinates_batch,
                "target_labels": target_labels_batch,
                "target_features": target_features_batch,
                "target_common_index": target_common_index,
                "target_idx": target_idx,
                "s2t_masked_coordinates": masked_coordinates_batch,
                "s2t_masked_labels": masked_labels_batch,
                "s2t_masked_features": masked_features_batch,
                "s2t_masked_idx": masked_idx,
                # "s2t_masked_mask": masked_mask,
                "s2t_masked_source_mask": source_masked_mask,
                "s2t_masked_target_mask": target_masked_mask}

