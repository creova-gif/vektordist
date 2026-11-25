terraform {
  required_version = ">= 1.5.0"

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }
}

variable "project_id" {
  description = "GCP project ID hosting the VektorDist benchmark cluster"
  type        = string
}

variable "region" {
  description = "GCP region for the benchmark VPC and persistent SSDs"
  type        = string
  default     = "us-central1"
}

variable "zone" {
  description = "Primary GCP zone for the 5-node VektorDist cluster"
  type        = string
  default     = "us-central1-a"
}

variable "node_count" {
  description = "Number of VektorDist shard replica nodes"
  type        = number
  default     = 5
}

variable "machine_type" {
  description = "Compute Engine machine type with AVX2/NEON SIMD support"
  type        = string
  default     = "n2-standard-8"
}

provider "google" {
  project = var.project_id
  region  = var.region
  zone    = var.zone
}

resource "google_compute_network" "vektordist_vpc" {
  name                    = "vektordist-bench-vpc"
  auto_create_subnetworks = false
}

resource "google_compute_subnetwork" "vektordist_subnet" {
  name          = "vektordist-bench-subnet"
  ip_cidr_range = "10.42.0.0/24"
  region        = var.region
  network       = google_compute_network.vektordist_vpc.id
}

resource "google_compute_firewall" "vektordist_internal_grpc" {
  name    = "vektordist-allow-internal-grpc"
  network = google_compute_network.vektordist_vpc.name

  allow {
    protocol = "tcp"
    ports    = ["50051", "50052"]
  }

  source_ranges = ["10.42.0.0/24"]
  target_tags   = ["vektordist-node", "vektordist-coord"]
}

resource "google_compute_disk" "vektordist_wal_ssd" {
  count = var.node_count
  name  = "vektordist-wal-ssd-${count.index + 1}"
  type  = "pd-ssd"
  zone  = var.zone
  size  = 100
}

resource "google_compute_instance" "vektordist_node" {
  count        = var.node_count
  name         = "vektordist-node-${count.index + 1}"
  machine_type = var.machine_type
  zone         = var.zone
  tags         = ["vektordist-node"]

  boot_disk {
    initialize_params {
      image = "debian-cloud/debian-12"
      size  = 30
      type  = "pd-balanced"
    }
  }

  attached_disk {
    source      = google_compute_disk.vektordist_wal_ssd[count.index].id
    device_name = "vektordist-wal-ssd"
  }

  network_interface {
    subnetwork = google_compute_subnetwork.vektordist_subnet.id
  }

  metadata_startup_script = <<-EOT
    #!/usr/bin/env bash
    set -euo pipefail
    mkdir -p /var/lib/vektordist/data
  EOT
}

output "node_internal_ips" {
  description = "Internal VPC IPv4 addresses of the 5 VektorDist shard replica nodes"
  value       = [for inst in google_compute_instance.vektordist_node : inst.network_interface[0].network_ip]
}
