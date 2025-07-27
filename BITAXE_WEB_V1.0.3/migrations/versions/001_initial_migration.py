"""Initial migration

Revision ID: 001
Revises: 
Create Date: 2024-07-27 10:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import sqlite

# revision identifiers, used by Alembic.
revision = '001'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Create logs table
    op.create_table('logs',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.Column('timestamp', sa.DateTime(), nullable=False),
        sa.Column('ip', sa.String(length=15), nullable=False),
        sa.Column('hostname', sa.String(length=50), nullable=True),
        sa.Column('temp', sa.Float(), nullable=True),
        sa.Column('hashRate', sa.Float(), nullable=True),
        sa.Column('power', sa.Float(), nullable=True),
        sa.Column('voltage', sa.Float(), nullable=True),
        sa.Column('frequency', sa.Integer(), nullable=True),
        sa.Column('coreVoltage', sa.Integer(), nullable=True),
        sa.Column('fanrpm', sa.Integer(), nullable=True),
        sa.Column('sharesAccepted', sa.Integer(), nullable=True),
        sa.Column('sharesRejected', sa.Integer(), nullable=True),
        sa.Column('uptime', sa.Integer(), nullable=True),
        sa.Column('version', sa.String(length=20), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_ip_timestamp', 'logs', ['ip', 'timestamp'], unique=False)
    op.create_index('idx_timestamp_desc', 'logs', ['timestamp'], unique=False)
    op.create_index(op.f('ix_logs_ip'), 'logs', ['ip'], unique=False)
    op.create_index(op.f('ix_logs_timestamp'), 'logs', ['timestamp'], unique=False)

    # Create tuning_status table
    op.create_table('tuning_status',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.Column('ip', sa.String(length=15), nullable=False),
        sa.Column('timestamp', sa.DateTime(), nullable=False),
        sa.Column('frequency', sa.Integer(), nullable=False),
        sa.Column('coreVoltage', sa.Integer(), nullable=False),
        sa.Column('avgHashRate', sa.Float(), nullable=True),
        sa.Column('avgTemperature', sa.Float(), nullable=True),
        sa.Column('avgEfficiency', sa.Float(), nullable=True),
        sa.Column('duration', sa.Integer(), nullable=True),
        sa.Column('failed', sa.Boolean(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_ip_efficiency', 'tuning_status', ['ip', 'avgEfficiency'], unique=False)
    op.create_index(op.f('ix_tuning_status_ip'), 'tuning_status', ['ip'], unique=False)

    # Create protocol table
    op.create_table('protocol',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.Column('timestamp', sa.DateTime(), nullable=False),
        sa.Column('ip', sa.String(length=15), nullable=False),
        sa.Column('event_type', sa.String(length=50), nullable=False),
        sa.Column('message', sa.Text(), nullable=True),
        sa.Column('severity', sa.String(length=20), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_ip_event_type', 'protocol', ['ip', 'event_type'], unique=False)
    op.create_index('idx_timestamp_desc_events', 'protocol', ['timestamp'], unique=False)
    op.create_index(op.f('ix_protocol_event_type'), 'protocol', ['event_type'], unique=False)
    op.create_index(op.f('ix_protocol_ip'), 'protocol', ['ip'], unique=False)
    op.create_index(op.f('ix_protocol_timestamp'), 'protocol', ['timestamp'], unique=False)

    # Create efficiency_markers table
    op.create_table('efficiency_markers',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.Column('timestamp', sa.DateTime(), nullable=False),
        sa.Column('ip', sa.String(length=15), nullable=False),
        sa.Column('efficiency', sa.Float(), nullable=False),
        sa.Column('hashRate', sa.Float(), nullable=True),
        sa.Column('power', sa.Float(), nullable=True),
        sa.Column('temperature', sa.Float(), nullable=True),
        sa.Column('frequency', sa.Integer(), nullable=True),
        sa.Column('coreVoltage', sa.Integer(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_ip_timestamp_eff', 'efficiency_markers', ['ip', 'timestamp'], unique=False)
    op.create_index(op.f('ix_efficiency_markers_ip'), 'efficiency_markers', ['ip'], unique=False)
    op.create_index(op.f('ix_efficiency_markers_timestamp'), 'efficiency_markers', ['timestamp'], unique=False)

    # Create benchmark_results table
    op.create_table('benchmark_results',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.Column('timestamp', sa.DateTime(), nullable=False),
        sa.Column('ip', sa.String(length=15), nullable=False),
        sa.Column('frequency', sa.Integer(), nullable=False),
        sa.Column('coreVoltage', sa.Integer(), nullable=False),
        sa.Column('averageHashRate', sa.Float(), nullable=True),
        sa.Column('averageTemperature', sa.Float(), nullable=True),
        sa.Column('efficiencyJTH', sa.Float(), nullable=True),
        sa.Column('averageVRTemp', sa.Float(), nullable=True),
        sa.Column('duration', sa.Integer(), nullable=True),
        sa.Column('samples_count', sa.Integer(), nullable=True),
        sa.Column('aborted', sa.Boolean(), nullable=True),
        sa.Column('abort_reason', sa.String(length=100), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_frequency_voltage', 'benchmark_results', ['frequency', 'coreVoltage'], unique=False)
    op.create_index('idx_hashrate_desc', 'benchmark_results', ['averageHashRate'], unique=False)
    op.create_index('idx_ip_efficiency_desc', 'benchmark_results', ['ip', 'efficiencyJTH'], unique=False)
    op.create_index(op.f('ix_benchmark_results_ip'), 'benchmark_results', ['ip'], unique=False)
    op.create_index(op.f('ix_benchmark_results_timestamp'), 'benchmark_results', ['timestamp'], unique=False)

    # Create miner_configurations table
    op.create_table('miner_configurations',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.Column('ip', sa.String(length=15), nullable=False),
        sa.Column('hostname', sa.String(length=50), nullable=True),
        sa.Column('alias', sa.String(length=50), nullable=True),
        sa.Column('color', sa.String(length=7), nullable=True),
        sa.Column('is_active', sa.Boolean(), nullable=True),
        sa.Column('location', sa.String(length=100), nullable=True),
        sa.Column('notes', sa.Text(), nullable=True),
        sa.Column('target_frequency', sa.Integer(), nullable=True),
        sa.Column('target_voltage', sa.Integer(), nullable=True),
        sa.Column('temp_limit', sa.Float(), nullable=True),
        sa.Column('temp_overheat', sa.Float(), nullable=True),
        sa.Column('autopilot_enabled', sa.Boolean(), nullable=True),
        sa.Column('benchmark_interval', sa.Integer(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('ip')
    )
    op.create_index(op.f('ix_miner_configurations_ip'), 'miner_configurations', ['ip'], unique=False)

    # Create system_configuration table
    op.create_table('system_configuration',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.Column('key', sa.String(length=100), nullable=False),
        sa.Column('value', sa.Text(), nullable=True),
        sa.Column('value_type', sa.String(length=20), nullable=True),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('category', sa.String(length=50), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('key')
    )
    op.create_index(op.f('ix_system_configuration_key'), 'system_configuration', ['key'], unique=False)


def downgrade() -> None:
    # Drop all tables in reverse order
    op.drop_table('system_configuration')
    op.drop_table('miner_configurations')
    op.drop_table('benchmark_results')
    op.drop_table('efficiency_markers')
    op.drop_table('protocol')
    op.drop_table('tuning_status')
    op.drop_table('logs')